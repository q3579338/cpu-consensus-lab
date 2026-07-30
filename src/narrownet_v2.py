#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet v2 —— 堵上 CoW 漏洞的窄深网络 PoW

v1 的漏洞:
    所有 nonce 共享同一个初始权重池, 每条链只改动 0.2% 的位置。
    攻击者可以用「写时复制」: 共享一份只读池 + 每链一张 1MB 的 delta 表,
    把每链内存从 256MB 压到 2MB, GPU 并行度暴涨 128 倍, 内存杠杆失效。

v2 的修法 (方案②: 每链池子从起点就不同):
    池子由 (seed, nonce) 共同决定, 且填充过程本身就是一条
    「顺序依赖 + 随机回读」的神经链。任意两个 nonce 的池子从第一个
    字节起就 100% 不同 -> 不存在可共享的基线 -> CoW 从根上失效。

避开的陷阱:
    如果填池只是「顺序写 256MB」, 那就是纯内存带宽活儿, 而 GPU 带宽
    是 CPU 的 10-20 倍, 等于把优势送给 GPU。所以填充必须:
      1. 顺序依赖   —— h[i] 依赖 h[i-1], 无法并行填
      2. 随机回读   —— 依赖已写入的随机位置, 无法预取/流水线化
      3. 神经形态   —— 与主链一致, 保持 AI 计算特征

两个阶段都是「窄深神经链」, 都严格串行, 都反 GPU。
"""

import hashlib
import sys
import time

import numpy as np


W = 8    # 宽度: 一个 AVX-512 FP64 向量


# ---------------- 参数预设 ----------------

PRESETS = {
    # 挖矿/服务端: 池子远超 L3, 逼出 DRAM 延迟, GPU 并行度被锁死在 显存/池子
    "mining":  dict(pool_mb=256, depth=1_048_576),
    # 浏览器/反爬: 64MB 是甜点 —— 手机能扛, 又能把 GPU 并行度压到 512 条
    "browser": dict(pool_mb=64,  depth=2_000_000),
    # 低端手机兜底
    "browser_lite": dict(pool_mb=16, depth=1_000_000),
    # 演示: 快速跑完
    "demo":    dict(pool_mb=2,   depth=8192),
}

# ---- 成本模型 (用于估算原生/WASM 真实耗时, Python 太慢无法直接测) ----
# 每步的工作: 8x8 FP64 矩阵向量乘(8条FMA) + 分支激活 + 归一化(含sqrt) + 1次随机访存
# 因为严格串行, 是 **延迟受限** 而非吞吐受限
CYCLES_PER_STEP = 55          # 保守估计: 计算~30周期 + 访存部分掩盖
NS_PER_STEP = {
    "cpu_avx512_5ghz":  CYCLES_PER_STEP / 5.0,      # ~11 ns  桌面CPU单核
    "wasm_simd128_3ghz": CYCLES_PER_STEP * 3 / 3.0,  # ~55 ns  WASM(128位SIMD,指令数x3~4)
    "mobile_wasm_2ghz":  CYCLES_PER_STEP * 4 / 2.0,  # ~110 ns 手机
    "gpu_fp64_serial":  500.0,                       # ~500ns 显存延迟, 串行链无法掩盖
}


# ---------------- 工具 ----------------

def _kdf(seed: bytes, n: int) -> bytes:
    out, i = bytearray(), 0
    while len(out) < n:
        out += hashlib.sha256(seed + i.to_bytes(8, "big")).digest()
        i += 1
    return bytes(out[:n])


def _to_unit(raw: bytes) -> np.ndarray:
    """字节流 -> [-1,1) 的 float64, 完全确定性。"""
    u = np.frombuffer(raw, dtype=np.uint64).copy()
    return (u >> 11).astype(np.float64) / float(1 << 53) * 2.0 - 1.0


def _fnv(h: np.ndarray) -> int:
    """由激活值导出 64 位索引种子 (指针追逐的来源)。"""
    acc = 0xcbf29ce484222325
    for b in h.view(np.uint64):
        acc = ((acc ^ int(b)) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return acc


def _act(h: np.ndarray) -> np.ndarray:
    """数据相关的分支非线性 (GPU 上造成 warp 发散) + 归一化。"""
    h = np.where(h > 0.0, h / (1.0 + np.abs(h)), 0.05 * h)
    return h / (np.sqrt(np.dot(h, h)) + 1e-12)


# ---------------- 阶段1: 顺序填池 (CoW 免疫) ----------------

def fill_pool(seed: bytes, nonce: int, n_entries: int):
    """
    用一条顺序神经链填满权重池。

    关键性质:
      * 起点 h 由 (seed, nonce) 决定 -> 每个 nonce 的池子完全不同
      * h[i] 依赖 h[i-1]             -> 无法并行填充
      * 每步回读一个已写入的随机位置   -> 无法预取, 制造读写依赖
    """
    blocks = n_entries // W
    pool = np.empty(n_entries, dtype=np.float64)

    # 固定的小权重矩阵 A/b: 从 seed 派生, 只有 72 个 double, 常驻 L1
    mat = _to_unit(_kdf(seed + b"|A", (W * W + W) * 8))
    A, bb = mat[:W * W].reshape(W, W), mat[W * W:]

    h = _to_unit(_kdf(seed + b"|h0|" + nonce.to_bytes(8, "big"), W * 8))

    for i in range(blocks):
        h = _act(A @ h + bb)                 # 神经前向 (顺序依赖)

        if i > 0:                            # 随机回读已写入的块
            j = _fnv(h) % i
            h = _act(h + 0.25 * pool[j * W:(j + 1) * W])

        pool[i * W:(i + 1) * W] = h          # 写入

    return pool, h


# ---------------- 阶段2: 主链 (指针追逐) ----------------

def main_chain(pool: np.ndarray, h: np.ndarray, depth: int):
    """
    D 层严格串行前向。每层的权重位置由上一层激活决定 (指针追逐),
    且结果写回池子 (读-改-写依赖, 关掉预取)。
    """
    span = pool.size - W * W - W
    for _ in range(depth):
        idx = _fnv(h) % span
        Wt = pool[idx:idx + W * W].reshape(W, W)
        bb = pool[idx + W * W:idx + W * W + W]

        h = _act(Wt @ h + bb)                # 真·神经网络层 (FP64)
        pool[idx] = pool[idx] * 0.5 + h[0] * 0.5   # 写回
    return h


def narrownet(seed: bytes, nonce: int, pool_mb: int, depth: int) -> bytes:
    n_entries = (pool_mb * 1024 * 1024) // 8
    pool, h = fill_pool(seed, nonce, n_entries)
    h = main_chain(pool, h, depth)
    return hashlib.sha256(h.tobytes()).digest()


# ---------------- 验证 ----------------

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    P = PRESETS[name]
    pool_mb, depth = P["pool_mb"], P["depth"]
    n_entries = (pool_mb * 1024 * 1024) // 8

    print(f"=== NarrowNet v2 [{name}] ===")
    print(f"宽度 {W}  池子 {pool_mb} MB ({n_entries:,} 条目)  深度 {depth:,}\n")

    seed = b"narrownet-v2-seed"

    # --- 计时 ---
    t0 = time.perf_counter()
    pool0, h0 = fill_pool(seed, 0, n_entries)
    t1 = time.perf_counter()
    h_end = main_chain(pool0.copy(), h0.copy(), depth)
    t2 = time.perf_counter()

    fill_ms, chain_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
    print(f"阶段1 填池   : {fill_ms:9.1f} ms  ({n_entries//W:,} 步)")
    print(f"阶段2 主链   : {chain_ms:9.1f} ms  ({depth:,} 层)")
    print(f"合计         : {fill_ms+chain_ms:9.1f} ms\n")

    # --- ★ CoW 免疫性验证: 不同 nonce 的池子应当 ~100% 不同 ---
    pool_a, _ = fill_pool(seed, 0, n_entries)
    pool_b, _ = fill_pool(seed, 1, n_entries)
    same = int(np.count_nonzero(pool_a == pool_b))
    print(f"[★ CoW 免疫] nonce=0 与 nonce=1 的池子:")
    print(f"    相同条目: {same:,} / {n_entries:,}  ({same/n_entries*100:.4f}%)")
    print(f"    -> {'✅ 无可共享基线, CoW 失效' if same/n_entries < 0.001 else '❌ 仍可共享!'}")
    print(f"    (v1 的情况是 99.8% 相同 -> CoW 可把内存压到 1/128)\n")

    # --- 确定性 ---
    d1 = narrownet(seed, 7, pool_mb, depth)
    d2 = narrownet(seed, 7, pool_mb, depth)
    print(f"[确定性] 同 (seed,nonce) 两次结果一致: {d1 == d2}")

    # --- 雪崩 ---
    d3 = narrownet(seed, 8, pool_mb, depth)
    diff = sum(bin(a ^ b).count("1") for a, b in zip(d1, d3))
    print(f"[雪崩效应] nonce 改 1, 输出不同 bit: {diff}/256 (理想≈128)")

    # --- 数值稳定 ---
    print(f"[数值稳定] 末层范数: {np.sqrt(np.dot(h_end,h_end)):.6f} (应≈1)")

    # --- 串行性: 深度翻倍 -> 时间翻倍 ---
    t3 = time.perf_counter(); main_chain(pool0.copy(), h0.copy(), depth * 2)
    t4 = time.perf_counter()
    print(f"[串行性] 深度翻倍耗时比: {(t4-t3)*1000/chain_ms:.2f}x (≈2.0 为严格串行)")

    # --- 内存杠杆 ---
    print(f"\n--- 内存杠杆 (每条链独占 {pool_mb} MB) ---")
    for vram, dev in [(32, "RTX 5090 32GB"), (80, "H100 80GB"), (192, "B200 192GB")]:
        n = vram * 1024 // pool_mb
        print(f"  {dev:<16}: 最多 {n:>6,} 条并行")
    for ram, dev in [(64, "CPU 64GB"), (512, "CPU 512GB")]:
        print(f"  {dev:<16}: 最多 {ram*1024//pool_mb:>6,} 条并行 (内存便宜 10-20x)")


if __name__ == "__main__":
    main()
