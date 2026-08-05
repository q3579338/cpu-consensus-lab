#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自查:16 线程掉到 75% 到底是什么造成的?

这决定 C1 是否成立。两个互斥解释:
  (a) 全核降频 —— 机器固有,与工作负载无关 -> SAT 和对照都掉,C1 成立
  (b) 线程被塞进 SMT 兄弟核(16 线程只占 8 物理核) -> 测量口径错,整列作废

判别法:实测不同并发下的**有效频率**。
用已知延迟的依赖链(imul 3 周期 + xor 1 周期 = 4 周期/次)反推实际 GHz,
这是 narrownet_v3.py 里 measure_ghz 的同一手法。

  若 16 线程时频率降到 ~75% -> (a) 降频,C1 成立
  若频率基本不降但吞吐掉 25% -> (b) 或别的争用,C1 需重审
"""
import json, os, statistics, threading, time

import numpy as np
from numba import njit

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=True, nogil=True)
def lat_chain(n):
    """纯依赖链:imul(3周期) + xor(1周期) = 4 周期/次,不吃内存。"""
    x = np.uint64(1)
    for _ in range(n):
        x = ((x * np.uint64(6364136223846793005)) ^ np.uint64(0x9E3779B97F4A7C15)) & M64
    return x


def ghz_at(n_threads, n=40_000_000, rounds=3):
    """n_threads 个线程同时跑依赖链,返回每线程实测 GHz(取最好一轮)。"""
    lat_chain(1000)                      # 预热 JIT
    best = 0.0
    for _ in range(rounds):
        times = [0.0] * n_threads

        def worker(i):
            t0 = time.perf_counter()
            lat_chain(n)
            times[i] = time.perf_counter() - t0

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in ts: t.start()
        for t in ts: t.join()
        ghz = n * 4.0 / statistics.median(times) / 1e9
        best = max(best, ghz)
    return best


def main():
    print("=== 自查:16 线程的 75% 是降频还是 SMT 打包? ===")
    print("方法:纯依赖链反推有效频率(不吃内存,只受频率影响)\n")
    print(f"  {'线程':>5}{'实测GHz':>10}{'相对单线程':>12}{'预期吞吐效率':>14}")
    base = None
    rows = []
    for n in (1, 2, 4, 8, 16, 32):
        g = ghz_at(n)
        if base is None:
            base = g
        rel = g / base
        print(f"  {n:>5}{g:>10.2f}{rel:>11.0%}{rel:>13.0%}")
        rows.append({"threads": n, "ghz": g, "rel_freq": rel})

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "freq_7950x.json")
    with open(dest, "w") as f:
        json.dump(rows, f, indent=2)

    f16 = next(r["rel_freq"] for r in rows if r["threads"] == 16)
    f32 = next(r["rel_freq"] for r in rows if r["threads"] == 32)
    print(f"""
=== 判读 ===
  16 线程相对频率 {f16:.0%} —— 实测吞吐效率 SAT 76% / SHA 75%
  32 线程相对频率 {f32:.0%} —— 实测吞吐效率 SAT 52% / SHA 45%

  若"相对频率"≈"吞吐效率" -> 退化 = 全核降频(机器固有),
     则 SAT 与对照同步下滑就是预期行为,C1(CDCL 线性扩展)成立。
  若频率几乎不降 -> 退化另有来源,C1 需重审。
  注:32 线程时每个物理核跑 2 个线程,依赖链本就无法靠 SMT 提速,
     所以 32 那档的"频率"会被 SMT 共享稀释,属正常。""")


if __name__ == "__main__":
    main()
