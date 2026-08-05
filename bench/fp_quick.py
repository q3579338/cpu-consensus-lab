#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快测 Cadical 求解时的峰值内存增量,判断"SAT 不争用"是否只是实例太小。"""
import gc, json, os, random, threading, time

import psutil
from pysat.solvers import Cadical153

RATIO = 4.26
HERE = os.path.dirname(os.path.abspath(__file__))


def gen(n, seed):
    rng = random.Random(seed)
    return [[v if rng.random() < 0.5 else -v
             for v in rng.sample(range(1, n + 1), 3)]
            for _ in range(int(n * RATIO))]


def main():
    proc = psutil.Process()
    print("=== Cadical 求解期间峰值内存增量 ===", flush=True)
    print("(实验二的争用断崖在 >=4MB/线程)\n", flush=True)
    print(f"  {'n':>6}{'子句':>7}{'求解(s)':>10}{'峰值+MB':>10}   判读", flush=True)
    rows = []
    for n in (260, 400, 600):
        cls = gen(n, 2)
        gc.collect()
        base = proc.memory_info().rss
        peak = [base]
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    peak[0] = max(peak[0], proc.memory_info().rss)
                except Exception:
                    pass
                time.sleep(0.005)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t0 = time.perf_counter()
        with Cadical153(bootstrap_with=cls) as sol:
            sol.conf_budget(3_000_000)
            sol.solve_limited()
        dt = time.perf_counter() - t0
        stop.set()
        th.join(timeout=1)
        peak[0] = max(peak[0], proc.memory_info().rss)
        mb = (peak[0] - base) / 1048576
        tag = "远小于4MB" if mb < 2 else ("小于4MB" if mb < 4 else "达到4MB")
        print(f"  {n:>6}{int(n*RATIO):>7}{dt:>10.2f}{mb:>10.2f}   {tag}", flush=True)
        rows.append({"n": n, "clauses": int(n * RATIO), "solve_s": dt, "peak_mb": mb})

    with open(os.path.join(HERE, "..", "sat_footprint.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("""
判读:
  工作集远小于 4MB -> "SAT 线性扩展"很可能只是实例小,C1 必须限定条件
  工作集达到 4MB 仍线性 -> C1 成立,CDCL 确实不受内存带宽约束""", flush=True)


if __name__ == "__main__":
    main()
