#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet 原生性能标定 v2 —— 修正混合架构(P核/E核)带来的测量误差

v1 的问题:
  Core Ultra 5 125H 是 4P + 8E + 2LP-E 的混合架构, Windows 调度器会把线程
  在大小核间迁移, E 核比 P 核慢 2-3 倍 -> 同一函数测出 183 与 527 ns 两个值。
  另外笔记本长时间满载会降频。

v2 的做法:
  1. 线程绑定到 CPU0 (P核)
  2. 进程优先级提到 HIGH
  3. 每项测 5 遍取 **最小值** (测延迟的标准做法, 滤掉调度/中断干扰)
  4. 用已知延迟的依赖链反推真实运行频率
"""

import ctypes
import time
import numpy as np
from numba import njit

W = 8


# ---------------- 绑核 + 提优先级 ----------------

def pin_and_boost():
    k32 = ctypes.windll.kernel32
    ok_aff = k32.SetThreadAffinityMask(k32.GetCurrentThread(), ctypes.c_size_t(1))  # CPU0
    ok_pri = k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)              # HIGH
    return bool(ok_aff), bool(ok_pri)


# ---------------- 频率标定 ----------------

@njit(cache=True)
def _lat_chain(n):
    """纯依赖链: 每次迭代 = imul(3周期) + xor(1周期) = 4 周期。"""
    x = np.uint64(1)
    for _ in range(n):
        x = ((x * np.uint64(6364136223846793005)) ^ np.uint64(0x9E3779B97F4A7C15)) \
            & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x


def measure_ghz(reps=5):
    _lat_chain(1000)
    n = 30_000_000
    best = min((lambda: (lambda t0: (_lat_chain(n), time.perf_counter() - t0)[1])(time.perf_counter()))()
               for _ in range(reps))
    return n * 4.0 / best / 1e9


# ---------------- 主链 (无辅助函数, 全展开) ----------------

@njit(cache=True, fastmath=False)
def chain(pool, h, depth):
    span = np.uint64(pool.size - 72)
    a = h.copy()
    tmp = np.empty(W, dtype=np.float64)
    for _ in range(depth):
        acc = np.uint64(0xcbf29ce484222325)
        for j in range(W):
            acc = ((acc ^ np.uint64(int(abs(a[j]) * 4503599627370496.0)))
                   * np.uint64(0x100000001b3)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        idx = int(acc % span)

        for r in range(W):
            s = pool[idx + 64 + r]
            off = idx + r * W
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
        pool[idx] = pool[idx] * 0.5 + a[0] * 0.5
    return a[0]


# ---------------- 标定 ----------------

def bench(pool, depth, reps=5):
    h = np.array([0.31, -0.22, 0.47, -0.11, 0.63, -0.38, 0.19, -0.52])
    chain(pool, h.copy(), 500)                     # 预热 + 填缓存
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter()
        chain(pool, h.copy(), depth)
        best = min(best, time.perf_counter() - t0)
    return best * 1e9 / depth


def main():
    aff, pri = pin_and_boost()
    print("=== NarrowNet 原生标定 v2 (Numba/LLVM, 绑P核, 取最小值) ===")
    print(f"绑核CPU0: {'✅' if aff else '❌'}   高优先级: {'✅' if pri else '❌'}\n")

    ghz = measure_ghz()
    print(f"[实测频率] {ghz:.2f} GHz\n")

    print(f"{'池子':>8}{'ns/步':>10}{'周期/步':>10}{'相对2MB':>10}   说明")
    print("-" * 62)
    base = None
    rows = []
    for mb in (2, 8, 32, 64, 256):
        n = mb * 1024 * 1024 // 8
        pool = np.random.default_rng(7).random(n) * 2 - 1
        depth = 300_000 if mb <= 64 else 200_000
        ns = bench(pool, depth)
        cyc = ns * ghz
        if base is None:
            base = ns
        note = ("L2 命中" if mb <= 2 else "L3 命中" if mb <= 32 else "超出L3→DRAM")
        print(f"{mb:>6} MB{ns:>10.1f}{cyc:>10.1f}{ns/base:>9.2f}x   {note}")
        rows.append((mb, ns, cyc))
        del pool

    print(f"\n=== 对成本模型的修正 ===")
    print(f"{'':<26}{'原假设':>10}{'实测(64MB)':>14}{'修正倍数':>10}")
    print("-" * 62)
    ns64 = [r[1] for r in rows if r[0] == 64][0]
    cyc64 = [r[2] for r in rows if r[0] == 64][0]
    print(f"{'周期/步':<26}{55:>10}{cyc64:>14.0f}{cyc64/55:>9.1f}x")
    print(f"{'ns/步 (原生CPU)':<26}{11.0:>10.1f}{ns64:>14.1f}{ns64/11.0:>9.1f}x")

    print(f"""
⚠️ 本机为 Core Ultra 5 125H (Meteor Lake), **无 AVX-512**, 只有 AVX2(4×FP64)。
   且 Numba 生成的代码不如手写 intrinsics。有 AVX-512 的桌面 CPU 应更快,
   但差距主要在计算部分, 而实测显示瓶颈是**访存延迟**, 所以提升有限。
""")


if __name__ == "__main__":
    main()
