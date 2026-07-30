#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet 原生性能标定 —— 用 Numba(LLVM JIT) 编译出真实机器码, 校准成本模型。

目的: 之前分析里的全部数字都建立在「55 周期/步」这个**未经验证的假设**上。
      这里实测出真实的 ns/步 和 周期/步。

注意:
  * fastmath=False 是**强制**的 —— PoW 要求逐位确定性, 不能让编译器
    重排浮点运算 (FMA 合并会改变舍入结果)。
  * 本机 Intel Core Ultra 5 125H (Meteor Lake) **无 AVX-512**, 只有 AVX2
    (256位 = 4个FP64)。测出的是主流消费级 CPU 的数字。
"""

import time
import numpy as np
from numba import njit

W = 8


# ---------------- 频率标定 ----------------

@njit(cache=True)
def _calib_chain(n):
    """依赖链: imul 延迟 3 周期。用于反推实际运行频率。"""
    x = np.uint64(1)
    for _ in range(n):
        x = (x * np.uint64(6364136223846793005) + np.uint64(1)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x


def measure_ghz():
    _calib_chain(1000)                      # 预热编译
    n = 50_000_000
    t0 = time.perf_counter()
    _calib_chain(n)
    dt = time.perf_counter() - t0
    # 每次迭代 = 1条imul(3周期) + 1条add(1周期), 依赖链 => ~4周期
    return n * 4.0 / dt / 1e9


# ---------------- 核心算子 ----------------

@njit(cache=True, fastmath=False, inline='always')
def _fnv(h_u):
    acc = np.uint64(0xcbf29ce484222325)
    for i in range(W):
        acc = ((acc ^ h_u[i]) * np.uint64(0x100000001b3)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return acc


@njit(cache=True, fastmath=False, inline='always')
def _act(h):
    """分支非线性 + 归一化。就地修改。"""
    s = 0.0
    for i in range(W):
        v = h[i]
        if v > 0.0:
            v = v / (1.0 + abs(v))
        else:
            v = 0.05 * v
        h[i] = v
        s += v * v
    inv = 1.0 / (np.sqrt(s) + 1e-12)
    for i in range(W):
        h[i] *= inv


@njit(cache=True, fastmath=False)
def fill_pool(pool, A, bb, h, blocks):
    """阶段1: 顺序填池。h[i] 依赖 h[i-1] + 随机回读已写块。"""
    tmp = np.empty(W, dtype=np.float64)
    h_u = h.view(np.uint64)
    for i in range(blocks):
        # 神经前向 A@h + b
        for r in range(W):
            acc = bb[r]
            for c in range(W):
                acc += A[r, c] * h[c]
            tmp[r] = acc
        for r in range(W):
            h[r] = tmp[r]
        _act(h)

        if i > 0:                              # 随机回读 (指针追逐)
            j = int(_fnv(h_u) % np.uint64(i))
            base = j * W
            for r in range(W):
                h[r] = h[r] + 0.25 * pool[base + r]
            _act(h)

        base = i * W                           # 写入
        for r in range(W):
            pool[base + r] = h[r]


@njit(cache=True, fastmath=False)
def main_chain(pool, h, depth):
    """阶段2: 主链。权重位置由上层激活决定, 结果写回池子。"""
    span = pool.size - W * W - W
    tmp = np.empty(W, dtype=np.float64)
    h_u = h.view(np.uint64)
    for _ in range(depth):
        idx = int(_fnv(h_u) % np.uint64(span))
        for r in range(W):                     # W_t @ h + b_t
            acc = pool[idx + W * W + r]
            off = idx + r * W
            for c in range(W):
                acc += pool[off + c] * h[c]
            tmp[r] = acc
        for r in range(W):
            h[r] = tmp[r]
        _act(h)
        pool[idx] = pool[idx] * 0.5 + h[0] * 0.5   # 写回


# ---------------- 标定 ----------------

def bench(pool_mb, depth, label):
    n_entries = pool_mb * 1024 * 1024 // 8
    blocks = n_entries // W

    rng = np.random.default_rng(42)
    pool = np.zeros(n_entries, dtype=np.float64)
    A = rng.random((W, W)) * 2 - 1
    bb = rng.random(W) * 2 - 1
    h = rng.random(W) * 2 - 1

    t0 = time.perf_counter()
    fill_pool(pool, A, bb, h.copy(), blocks)
    t1 = time.perf_counter()
    h2 = h.copy()
    main_chain(pool, h2, depth)
    t2 = time.perf_counter()

    fill_ns = (t1 - t0) * 1e9 / blocks
    chain_ns = (t2 - t1) * 1e9 / depth
    print(f"  {label:<22} 填池 {fill_ns:6.1f} ns/步   主链 {chain_ns:6.1f} ns/步   "
          f"总 {(t2-t0)*1000:7.1f} ms")
    return fill_ns, chain_ns


def main():
    print("=== NarrowNet 原生性能标定 (Numba/LLVM JIT) ===\n")

    print("[预热编译中...]")
    warm = np.zeros(8192, dtype=np.float64)
    A = np.ones((W, W)); bb = np.ones(W); h = np.ones(W)
    fill_pool(warm, A, bb, h.copy(), 64)
    main_chain(warm, h.copy(), 64)

    ghz = measure_ghz()
    print(f"[实测运行频率] {ghz:.2f} GHz\n")

    print("池子大小对访存延迟的影响 (深度固定 200k):")
    results = {}
    for mb in (2, 16, 64, 256):
        results[mb] = bench(mb, 200_000, f"池子 {mb:>3} MB")

    print(f"\n=== 与成本模型对比 ===")
    print(f"{'池子':<10}{'主链 ns/步':>12}{'实测周期/步':>14}{'模型假设':>12}{'偏差':>10}")
    print("-" * 58)
    for mb, (_, chain_ns) in results.items():
        cyc = chain_ns * ghz
        print(f"{mb:>4} MB {chain_ns:>13.1f}{cyc:>14.1f}{55:>12}{cyc/55:>9.2f}x")

    print(f"""
说明:
  * 池子越大, 主链 ns/步 越高 -> 因为随机访问从 L2/L3 掉到 DRAM。
    这正是「内存杠杆」在起作用: 池子超过缓存, 每步都要吃 DRAM 延迟。
  * 本机无 AVX-512 (Meteor Lake 只有 AVX2), 有 AVX-512 的机器应更快。
  * Numba 生成的是 LLVM 优化过的机器码, 但不如手写 intrinsics;
    真实的 C/AVX-512 实现应该还能再快一些, 所以这是**保守下界**。
""")


if __name__ == "__main__":
    main()
