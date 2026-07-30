#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程缓存争用实验 —— 量化「小机器优势」的 M1 机制

问题: 同一台机器上多开 N 个挖矿线程, 每线程的速度会退化多少?

  * 若退化很小 (总吞吐 ≈ N x 单线程)  -> 大机器线性划算, 小机器无优势
  * 若退化严重 (总吞吐远小于 N x)      -> **这就是小机器优势的来源**
    因为 N 台单核小机器各有独立的缓存/内存子系统, 不互相踩踏。

对比不同工作集大小, 找出「让大机器最难受」的那个尺寸。
"""

import ctypes
import threading
import time

import numpy as np
from numba import njit

W = 8
BLK = 64
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
TO_INT = 4503599627370496.0


@njit(cache=True, fastmath=False, nogil=True)      # nogil: 真并行
def chain(pool, h, depth, blk_mask):
    a = h.copy()
    tmp = np.empty(W, dtype=np.float64)
    for _ in range(depth):
        acc = np.uint64(0xcbf29ce484222325)
        for j in range(W):
            acc = ((acc ^ np.uint64(int(abs(a[j]) * TO_INT))) * np.uint64(0x100000001b3)) & M64
        base = int(acc & blk_mask) * BLK
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
        pool[base] = pool[base] * 0.5 + a[0] * 0.5
    return a[0]


def run_n_threads(n_threads, pool_kb, depth):
    """每个线程持有**自己的**工作集(模拟独立矿工进程)。"""
    n_entries = pool_kb * 1024 // 8
    blk_mask = np.uint64((n_entries // BLK) - 1)
    pools = [np.random.default_rng(i).random(n_entries) * 2 - 1 for i in range(n_threads)]
    hs = [np.linspace(-0.5 + i * 0.01, 0.5, W) for i in range(n_threads)]

    for i in range(n_threads):                      # 预热 + 填缓存
        chain(pools[i], hs[i].copy(), 300, blk_mask)

    times = [0.0] * n_threads

    def worker(i):
        t0 = time.perf_counter()
        chain(pools[i], hs[i].copy(), depth, blk_mask)
        times[i] = time.perf_counter() - t0

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0

    per_thread_ns = (sum(times) / n_threads) * 1e9 / depth
    total_throughput = n_threads * depth / wall / 1e6      # M步/秒
    return per_thread_ns, total_throughput


def main():
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)

    import os
    ncpu = os.cpu_count()
    print(f"=== 多线程缓存争用实验 ===")
    print(f"逻辑核心 {ncpu} 个 (Core Ultra 5 125H: 4P + 8E + 2LP-E)\n")

    depth = 120_000
    for pool_kb, label in ((256, "256 KB (贴合 L2)"),
                           (2048, "2 MB (贴合 L2/L3)"),
                           (16384, "16 MB (吃 L3)")):
        print(f"--- 工作集/线程 = {label} ---")
        print(f"  {'线程数':>6}{'ns/步/线程':>13}{'总吞吐(M步/s)':>16}{'单线程退化':>12}{'扩展效率':>10}")
        base_ns = None
        base_tp = None
        for n in (1, 2, 4, 8, 12):
            ns, tp = run_n_threads(n, pool_kb, depth)
            if base_ns is None:
                base_ns, base_tp = ns, tp
            degrade = ns / base_ns
            eff = (tp / base_tp) / n          # 1.0=完美线性扩展
            flag = "✅" if eff > 0.8 else ("⚠️" if eff > 0.5 else "❌")
            print(f"  {n:>6}{ns:>13.1f}{tp:>16.1f}{degrade:>11.2f}x{eff:>9.0%} {flag}")
        print()

    print("""解读:
  扩展效率 = (N线程总吞吐 / 单线程吞吐) / N
    接近 100% -> 大机器线性划算, **小机器没有优势**
    远低于100% -> 多开互相踩踏, **N 台单核小机器胜出** <- 这就是 M1 机制
  找出扩展效率最低的工作集尺寸, 那就是「最偏袒小机器」的参数。""")


if __name__ == "__main__":
    main()
