#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补一个**同口径的真·零内存对照**。

为什么需要:第一版用 CPython 的 SHA-256 循环当"工作集≈0"对照,但那个循环
每次迭代都分配新 bytes 对象、做引用计数,内存行为并不可忽略。而频率自查显示
16 线程时频率仍有 98%,却测出 75% 吞吐——说明损失既不是降频,也不该全算在
"机器固有上限"头上。对照本身不干净,"SAT 贴合对照"这个论证就站不住。

这里用**纯 numba 依赖链**(不分配、不访存,只有寄存器上的 imul+xor)作对照,
且**用与 SAT 完全相同的多进程 harness**(mp.Pool,每进程同样工作量,取最小值),
消除 线程vs进程 的口径差异。

得到的才是这台机器"纯计算能扩展到多少"的真实上限。
"""
import json, multiprocessing as mp, os, statistics, time

COUNTS = [1, 2, 4, 8, 16, 32]
HERE = os.path.dirname(os.path.abspath(__file__))


def pure_worker(n_iter):
    """纯寄存器依赖链:imul(3周期)+xor(1周期),零内存访问、零分配。"""
    import numpy as np
    from numba import njit
    M64 = np.uint64(0xFFFFFFFFFFFFFFFF)

    @njit(cache=True)
    def chain(n):
        x = np.uint64(1)
        for _ in range(n):
            x = ((x * np.uint64(6364136223846793005))
                 ^ np.uint64(0x9E3779B97F4A7C15)) & M64
        return x

    chain(1000)                      # JIT 预热(不计时)
    t0 = time.perf_counter()
    chain(n_iter)
    return time.perf_counter() - t0


def run(n_procs, n_iter, rounds):
    best = None
    for _ in range(rounds):
        with mp.Pool(processes=n_procs) as pool:
            pool.map(pure_worker, [1000])                 # 预热池
            times = pool.map(pure_worker, [n_iter] * n_procs)
        per = statistics.median(times)
        best = per if best is None else min(best, per)
    return best


def main():
    print("=== 真·零内存对照(纯计算,与 SAT 同口径:多进程)===", flush=True)
    print("对照体:寄存器依赖链,无访存无分配\n", flush=True)
    n_iter = 120_000_000
    rounds = 3
    print(f"  {'并发':>5}{'单worker(s)':>13}{'退化':>8}{'扩展效率':>10}", flush=True)
    t1 = None
    rows = []
    for n in COUNTS:
        t = run(n, n_iter, rounds)
        if t1 is None:
            t1 = t
        eff = t1 / t
        print(f"  {n:>5}{t:>13.3f}{t/t1:>7.2f}x{eff:>9.0%}", flush=True)
        rows.append({"threads": n, "per_worker_s": t, "efficiency": eff})

    with open(os.path.join(HERE, "..", "true_baseline_7950x.json"), "w") as f:
        json.dump(rows, f, indent=2)

    e = {r["threads"]: r["efficiency"] for r in rows}
    print(f"""
=== 三方对比 @ 16 线程(=满物理核)===
  纯计算(真基线)     {e.get(16,0):.0%}
  SAT (Cadical, 5.2MB)  76%
  NarrowNet 链 (4MB)    33%

  纯计算 -> SAT 的差 = SAT 真实承受的争用
  SAT -> NarrowNet 的差 = 刻意设计出来的额外争用""", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
