#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存争用复测 —— 在同构硅上重跑,替换被污染的原始数据

README 自陈:原 12 线程那列「被大小核调度和笔记本温度墙放大了,需在同构服务器上重测」。
本机 7950X 正合适:Zen4 纯同构 16C/32T,无 P/E 核。

与原版 contention.py 的唯一区别:线程档位扩到 32,工作集加了 32MB 档,
计算核心 chain() 逐字未改 —— 保证数字可直接和原表对比。

7950X 缓存拓扑(决定预期拐点):
  L1d 32KB/核 · L2 1MB/核 · L3 32MB/CCD x 2 CCD(不跨 CCD 共享)
  -> 1-8 线程落单 CCD(共享 32MB L3);16 线程跨双 CCD;32 线程吃满 SMT
  -> 每线程 >4MB 时,8 线程就该开始踩踏(8x4=32MB 正好填满一个 CCD 的 L3)
"""
import ctypes, json, os, threading, time

import numpy as np
from numba import njit

W = 8
BLK = 64
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
TO_INT = 4503599627370496.0

THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
WORKSETS = [(256, "256 KB (贴合 L2)"),
            (2048, "2 MB (原版甜点)"),
            (4096, "4 MB (8线程=32MB=单CCD L3)"),
            (16384, "16 MB (吃 L3)")]


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
    """每线程持有**自己的**工作集(模拟独立矿工进程)。"""
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
    try:                                            # 提高优先级,减少调度噪声
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass                                        # 非 Windows 忽略

    ncpu = os.cpu_count()
    print("=== 多线程缓存争用复测(同构硅)===")
    print(f"CPU: AMD Ryzen 9 7950X | {ncpu} 逻辑核 = 16 物理核 x SMT, 纯同构")
    print("L3: 32MB/CCD x 2(不跨 CCD 共享)\n")

    depth = 120_000
    results = {}
    for pool_kb, label in WORKSETS:
        print(f"--- 工作集/线程 = {label} ---")
        print(f"  {'线程数':>6}{'ns/步/线程':>13}{'总吞吐(M步/s)':>16}{'单线程退化':>12}{'扩展效率':>10}")
        base_ns = base_tp = None
        rows = []
        for n in THREAD_COUNTS:
            ns, tp = run_n_threads(n, pool_kb, depth)
            if base_ns is None:
                base_ns, base_tp = ns, tp
            degrade = ns / base_ns
            eff = (tp / base_tp) / n
            flag = "OK" if eff > 0.8 else ("warn" if eff > 0.5 else "BAD")
            print(f"  {n:>6}{ns:>13.1f}{tp:>16.1f}{degrade:>11.2f}x{eff:>9.0%} {flag}")
            rows.append({"threads": n, "ns_per_step": ns, "throughput_Msteps": tp,
                         "degrade": degrade, "efficiency": eff})
        results[label] = rows
        print()

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "contention_7950x.json")
    with open(dest, "w") as f:
        json.dump({"cpu": "AMD Ryzen 9 7950X", "logical_cores": ncpu,
                   "depth": depth, "results": results}, f, indent=2)
    print(f"结果 -> {os.path.normpath(dest)}")
    print("""
解读:
  扩展效率 = (N线程总吞吐 / 单线程吞吐) / N
    接近 100% -> 大机器线性划算,**小机器无优势**
    远低于100% -> 多开互相踩踏,**N 台单核小机器胜出**(这就是设计想要的杠杆)
  重点看 8->16 这一跳:若那里断崖,说明跨 CCD / L3 容量是真实边界。""")


if __name__ == "__main__":
    main()
