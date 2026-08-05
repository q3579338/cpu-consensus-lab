#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解释一个反直觉的消融结果:**去掉写回,GPU 反而慢了 25–44%**。

假设:两个变体并不是"同样的活减一条 store"。写回会改池值 -> 改下一步激活
-> 改块地址,**两者走的是完全不同的地址轨迹**。若有写回那版的轨迹更扎堆
(局部性更好),它就会更快 —— 那么 v3 里这条"用来废掉 GPU 只读缓存"的写回,
实际效果可能与设计意图相反:**它在帮 GPU,而不是害 GPU**。

本脚本直接量轨迹的局部性,不再靠推测:
  * 触碰到的**不同块**数(越少越扎堆)
  * 重复率 = 1 - 不同块/总步数
  * 地址分布的归一化熵(1.0 = 均匀撒开,越小越扎堆)
  * 相邻两步地址差的中位数(小 = 空间局部性好)
"""
import math
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.narrownet_v3 import BLK, M64, TO_INT, W  # noqa: E402

DEPTH = 200_000


@njit(cache=True, fastmath=False)
def trace(pool, h, depth, blk_mask, writeback, out):
    """跑链并记录每步的块索引。writeback=1 复刻 v3;0 为只读变体。"""
    a = h.copy()
    tmp = np.empty(W, dtype=np.float64)
    for step in range(depth):
        acc = np.uint64(0xcbf29ce484222325)
        for j in range(W):
            acc = ((acc ^ np.uint64(int(abs(a[j]) * TO_INT))) * np.uint64(0x100000001b3)) & M64
        blk = int(acc & blk_mask)
        out[step] = blk
        base = blk * BLK
        for r in range(W):
            s = a[r]
            off = base + r * W
            for c in range(W):
                s += pool[off + c] * a[c]
            tmp[r] = s
        s2 = 0.0
        for r in range(W):
            v = tmp[r]
            v = v / (1.0 + abs(v)) if v > 0.0 else 0.05 * v
            a[r] = v
            s2 += v * v
        inv = 1.0 / (np.sqrt(s2) + 1e-12)
        for r in range(W):
            a[r] *= inv
        if writeback == 1:
            pool[base] = pool[base] * 0.5 + a[0] * 0.5
    return a


@njit(cache=True, fastmath=False)
def fill(pool, s0, s1, mask):
    n = pool.size
    x, y = s0, s1
    for i in range(n):
        t = x
        t ^= (t << np.uint64(23)) & M64
        t ^= t >> np.uint64(17)
        t ^= y ^ (y >> np.uint64(26))
        x, y = y, t
        v = (x + y) & M64
        j = int(v & mask)
        v = (v ^ np.uint64(int(abs(pool[j]) * TO_INT))) & M64
        v = (v * np.uint64(0x9E3779B97F4A7C15)) & M64
        pool[i] = (v >> np.uint64(11)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0


def stats(seq, n_blocks):
    uniq, counts = np.unique(seq, return_counts=True)
    p = counts / counts.sum()
    ent = -(p * np.log2(p)).sum()
    ent_max = math.log2(len(seq))          # 全不重复时的熵上限
    jumps = np.abs(np.diff(seq.astype(np.int64)))
    return {
        "distinct": len(uniq),
        "repeat_rate": 1.0 - len(uniq) / len(seq),
        "entropy_norm": ent / ent_max,
        "median_jump_blocks": float(np.median(jumps)),
        "top1_share": float(counts.max() / counts.sum()),
    }


def main():
    print("=== 写回如何改变地址轨迹的局部性 ===")
    print(f"    depth={DEPTH:,};归一化熵 1.0=均匀撒开,越小越扎堆\n")
    print(f"  {'池子':>6}{'变体':>7}{'不同块':>9}{'重复率':>9}{'归一熵':>9}"
          f"{'中位跳距':>11}{'最热块占比':>11}")
    print("-" * 64)
    for mb in (64, 256):
        n = (mb * 1024 * 1024) // 8
        n_blocks = n // BLK
        pool0 = np.zeros(n, dtype=np.float64)
        fill(pool0, np.uint64(0x243F6A8885A308D3), np.uint64(0x13198A2E03707345),
             np.uint64(n - 1))
        h = np.linspace(-0.5, 0.5, W)
        res = {}
        for wb, name in ((1, "有写回"), (0, "无写回")):
            out = np.zeros(DEPTH, dtype=np.int64)
            trace(pool0.copy(), h.copy(), DEPTH, np.uint64(n_blocks - 1), wb, out)
            st = stats(out, n_blocks)
            res[name] = st
            print(f"  {mb:>4}MB{name:>7}{st['distinct']:>9,}{st['repeat_rate']:>8.1%}"
                  f"{st['entropy_norm']:>9.3f}{st['median_jump_blocks']:>11,.0f}"
                  f"{st['top1_share']:>10.2%}")
        d = res["有写回"]["distinct"] - res["无写回"]["distinct"]
        print(f"        -> 有写回比无写回多碰 {d:+,} 个不同块\n")
        del pool0

    print("""判读:
  若两版的不同块数/熵**基本一致** -> 局部性不是原因,GPU 变慢另有来源
     (最可能是编译器调度:store 消失后寄存器分配/指令调度变了),
     则"写回帮了 GPU"的说法不成立,消融结论需改口径。
  若有写回那版明显更扎堆 -> 写回确实在给 GPU 送局部性,
     那么它作为"反 GPU 手段"是**反效果的**,去掉它反而是纯赚。""")


if __name__ == "__main__":
    main()
