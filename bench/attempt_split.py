#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次 nonce 尝试由两段**都是串行**的工作组成,这里量出各占多少:

    一次尝试 = 填池(fill_pool,x[i] 依赖 x[i-1]) + 跑链(main_chain,深度 depth)

为什么要分开量:**验证成本 = 一次尝试的成本**,而验证者必须两段都重做。
所以填池那段是验证成本的**地板**——无论链调多短都省不掉。
这个地板决定了"把链改短来降低验证成本"这条路能走多远。
"""
import os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.narrownet_v3 import BLK, fill_pool, main_chain, measure_ghz, pin_and_boost  # noqa: E402

DEPTH = 1_000_000


def main():
    pin_and_boost(0)
    ghz = measure_ghz()
    print(f"=== 一次尝试的成本拆解(实测 {ghz:.2f} GHz)===")
    print("  一次尝试 = 填池(串行) + 跑链(串行);验证 = 把这两段都重做\n")

    # 预热 JIT(小池子、浅链)
    warm = np.zeros(1 << 16, dtype=np.float64)
    fill_pool(warm, np.uint64(1), np.uint64(3), np.uint64(warm.size - 1))
    main_chain(warm, np.linspace(-0.5, 0.5, 8), 1000, np.uint64((warm.size // BLK) - 1))

    print(f"{'池子':>7}{'填池(ms)':>11}{'链 100万步(ms)':>16}{'合计':>9}{'填池占比':>10}{'验证地板':>11}")
    print("-" * 66)
    for mb in (16, 64, 256, 1024):
        n = (mb * 1024 * 1024) // 8
        pool = np.zeros(n, dtype=np.float64)

        best_fill = 1e18
        for _ in range(3):
            t0 = time.perf_counter()
            fill_pool(pool, np.uint64(0x243F6A8885A308D3), np.uint64(0x13198A2E03707345),
                      np.uint64(n - 1))
            best_fill = min(best_fill, time.perf_counter() - t0)

        h = np.linspace(-0.5, 0.5, 8)
        best_chain = 1e18
        for _ in range(2):
            t0 = time.perf_counter()
            main_chain(pool, h.copy(), DEPTH, np.uint64((n // BLK) - 1))
            best_chain = min(best_chain, time.perf_counter() - t0)

        f_ms, c_ms = best_fill * 1e3, best_chain * 1e3
        tot = f_ms + c_ms
        print(f"{mb:>5}MB{f_ms:>11.1f}{c_ms:>16.1f}{tot:>9.0f}{f_ms/tot:>9.0%}"
              f"{f_ms:>10.1f}ms")
        del pool

    print("""
判读:
  "填池"那一列就是**验证成本的地板**——链再短也省不掉,验证者必须重填。
  填池占比越高,说明"缩短链来降低验证成本"的空间越小。

  注意:填池和跑链**两段都是串行的**(fill_pool 里 x[i] 依赖 x[i-1]),
  所以缩短链并不会把工作让给 GPU —— 这一点与我先前的判断相反,
  bench/narrownet_cuda.py 里两边都用并行 splitmix 填池、只比了链,
  因此那组 CPU/GPU 数字对 NarrowNet 是**保守**的。""")


if __name__ == "__main__":
    main()
