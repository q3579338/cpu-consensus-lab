#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet-PoW —— 窄而极深的神经网络工作量证明

设计目标: 计算形态是"真·神经网络前向传播"(AI 特征),
          但把每一个维度都调到 GPU 的反面。

                    常规 AI (GPU 主场)      NarrowNet (CPU 主场)
    宽度 W          4096+                   8   (= 一个 AVX-512 FP64 向量)
    深度 D          几十层                   65536 层
    batch           几千                     1
    数值精度        FP16 / BF16              FP64 严格 IEEE
    权重            常驻显存, 反复复用        每层从 32MB 池按上层激活随机取

GPU 崩溃的六个点:
    1. 宽度 8   -> 每层仅 64 次乘加, 填不满一个 warp(32线程), 万核闲置
    2. 层间串行 -> 深度方向无法并行, 堆核心无用
    3. launch   -> 每层一次 kernel launch ~5us, x65536 = 327ms 纯开销
    4. FP64     -> 消费级 GPU 的 FP64 吞吐是 FP32 的 1/64
    5. 指针追逐 -> 取哪块权重由上一层激活决定, 无法预取/合并访存
    6. 分支发散 -> 激活值决定非线性分支, warp 锁步被迫串行

CPU 反过来全占便宜: AVX-512 单指令 8xFP64、L1 4 周期、分支预测、5GHz 单核。
"""

import hashlib
import struct
import sys
import time

import numpy as np


# ---------------- 参数 ----------------

W = 8                 # 宽度: 刚好一个 AVX-512 FP64 向量

# 权重池大小。关键约束: 必须 **超过 CPU 的 L3** (现代 CPU L3 达 32-64MB),
# 否则整个池装进 L3, "内存延迟"这根反 GPU 的杠杆就废了。
# 超过 L3 后每层的随机取权重都要吃 DRAM 延迟:
#     CPU  ~80 ns   (乱序执行 + 预取器仍能部分掩盖)
#     GPU  ~400-800 ns, 且串行链上无并行线程可切换来掩盖  <- 差距在这
POOL_LOG2 = 25        # 2^25 个 float64 = 256 MB
POOL = 1 << POOL_LOG2


# ---------------- 初始化 ----------------

def _kdf(seed: bytes, nbytes: int) -> bytes:
    """从种子确定性扩展出任意长度字节流 (SHA256 计数器模式)。"""
    out, i = bytearray(), 0
    while len(out) < nbytes:
        out += hashlib.sha256(seed + i.to_bytes(8, "big")).digest()
        i += 1
    return bytes(out[:nbytes])


def build_pool(seed: bytes) -> np.ndarray:
    """
    生成 32MB 权重池。值域压到 [-1,1) 附近, 避免前向传播数值爆炸。
    这一步本身也要占满内存带宽, GPU 没便宜可占。
    """
    raw = _kdf(seed + b"|pool", POOL * 8)
    u = np.frombuffer(raw, dtype=np.uint64).copy()
    # 映射到 [-1, 1): 取 53 位尾数做定点转浮点, 完全确定性
    m = (u >> 11).astype(np.float64) / float(1 << 53)   # [0,1)
    return (m * 2.0 - 1.0)                               # [-1,1)


def init_activation(seed: bytes) -> np.ndarray:
    raw = _kdf(seed + b"|act", W * 8)
    u = np.frombuffer(raw, dtype=np.uint64).copy()
    m = (u >> 11).astype(np.float64) / float(1 << 53)
    return m * 2.0 - 1.0


# ---------------- 核心: 前向传播链 ----------------

def _idx_from_activation(h: np.ndarray) -> int:
    """
    由当前激活确定性地导出权重池索引 —— 这就是"指针追逐"。
    取 float64 的原始位, 混合成一个索引。GPU 无法预取这个地址。
    """
    acc = 0xcbf29ce484222325                      # FNV-1a 64 位
    for b in h.view(np.uint64):
        acc = ((acc ^ int(b)) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return acc % (POOL - W * W - W)


def forward_chain(pool: np.ndarray, h: np.ndarray, D: int, writeback: bool = True):
    """
    D 层严格串行的前向传播。
    每层:
      1. 由 h 导出 idx            (指针追逐, 地址不可预测)
      2. 从池里取 8x8 权重 + 8 偏置
      3. h = Wt @ h + b           (真·神经网络层)
      4. 激活值相关的分支非线性     (制造 warp 发散)
      5. 归一化                    (防爆炸, 且强化串行依赖)
      6. 把结果写回权重池          (读写依赖, 彻底关掉预取)
    """
    for _ in range(D):
        idx = _idx_from_activation(h)

        Wt = pool[idx: idx + W * W].reshape(W, W)
        b = pool[idx + W * W: idx + W * W + W]

        h = Wt @ h + b                       # 8x8 矩阵 x 8 向量, FP64

        # --- 数据相关分支: GPU 上会让整个 warp 走两条路 ---
        pos = h > 0.0
        h = np.where(pos,
                     h / (1.0 + np.abs(h)),   # 软饱和 (正半轴)
                     0.05 * h)                # leaky (负半轴)

        # --- 归一化: 保持数值稳定, 同时让每层都依赖全部分量 ---
        n = np.sqrt(np.dot(h, h)) + 1e-12
        h = h / n

        # --- 写回权重池: 制造读-改-写依赖 ---
        if writeback:
            pool[idx] = pool[idx] * 0.5 + h[0] * 0.5

    return h


def narrownet_pow(seed: bytes, D: int) -> bytes:
    """完整一轮: 建池 -> 深链前向 -> 摘要。"""
    pool = build_pool(seed)
    h = init_activation(seed)
    h = forward_chain(pool, h, D)
    return hashlib.sha256(h.tobytes()).digest()


# ---------------- 演示 ----------------

def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 4096

    print(f"=== NarrowNet-PoW ===")
    print(f"宽度 W = {W}   深度 D = {D:,}   权重池 = {POOL*8/1024/1024:.0f} MB\n")

    seed = b"narrownet-demo-seed-0001"

    t0 = time.perf_counter()
    pool = build_pool(seed)
    t1 = time.perf_counter()
    print(f"建权重池      : {(t1-t0)*1000:8.1f} ms  ({POOL*8/1024/1024:.0f} MB)")

    h0 = init_activation(seed)
    t2 = time.perf_counter()
    h = forward_chain(pool.copy(), h0.copy(), D)
    t3 = time.perf_counter()
    fw = (t3 - t2) * 1000
    print(f"前向链 {D} 层 : {fw:8.1f} ms  ({fw*1000/D:.2f} µs/层)")

    digest = hashlib.sha256(h.tobytes()).digest()
    print(f"输出摘要      : {digest.hex()[:32]}...")

    # --- 确定性验证: 同种子必须逐位一致 ---
    d2 = narrownet_pow(seed, D)
    d1 = hashlib.sha256(forward_chain(build_pool(seed), init_activation(seed), D).tobytes()).digest()
    print(f"\n[=] 确定性(同种子两次结果一致): {d1 == d2}")

    # --- 雪崩效应: 种子改 1 bit, 输出应完全不同 ---
    d3 = narrownet_pow(b"narrownet-demo-seed-0002", D)
    diff = sum(bin(a ^ b).count("1") for a, b in zip(d2, d3))
    print(f"[=] 雪崩效应(改种子后不同 bit 数): {diff}/256  (理想≈128)")

    # --- 数值稳定性: 激活不能爆炸或塌缩 ---
    print(f"[=] 末层激活范数: {np.sqrt(np.dot(h,h)):.6f}  (归一化后应≈1)")
    print(f"[=] 末层激活范围: [{h.min():.4f}, {h.max():.4f}]")

    # --- 串行性检验: 深度翻倍, 时间应线性翻倍 ---
    t4 = time.perf_counter()
    forward_chain(pool.copy(), h0.copy(), D * 2)
    t5 = time.perf_counter()
    fw2 = (t5 - t4) * 1000
    print(f"\n[=] 深度翻倍耗时比: {fw2/fw:.2f}x  (≈2.0 说明严格线性串行, 无并行捷径)")

    # --- GPU 劣势量化 ---
    launch_us = 5.0
    gpu_launch_ms = D * launch_us / 1000
    print(f"\n--- GPU 侧成本估算 (D={D:,}) ---")
    print(f"仅 kernel launch 开销 : {gpu_launch_ms:8.1f} ms  (每层 {launch_us}µs)")
    print(f"CPU 实测全程          : {fw:8.1f} ms")
    print(f"-> GPU 光启动开销就是 CPU 全程的 {gpu_launch_ms/fw:.2f}x, 还没开始算")
    print(f"   再叠加: 宽度8填不满warp / FP64降速64x / 指针追逐 / 分支发散")


if __name__ == "__main__":
    main()
