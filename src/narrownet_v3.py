#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet v3 —— 按实测标定结果修正

v2 实测暴露的问题:
  1. 成本模型错 15x: 漏算了 8 次 FP 除法(50-110周期) 和 64位取模(30-90周期)
  2. 填池太贵: 64MB 池子光填充就要 894ms(原生), 浏览器场景不可用
  3. 绑核失败: ctypes 类型没声明, SetThreadAffinityMask 静默返回 0

v3 的修正:
  1. 池子大小取 2 的幂 -> `% span` 变成 `& mask`, 每步省 30-90 周期
  2. 填池换成**廉价的整数顺序链** —— 填池只需"不可并行、不可共享",
     不需要昂贵。昂贵的神经计算留给有指针追逐的主链。
  3. 正确声明 ctypes 类型, 真正绑到 P 核

设计改进 (顺带):
  去掉偏置, 改用**残差连接** h <- act(W·h + h):
    * 权重块正好 64 个 double = 512 字节 = 8 个缓存行, 2 的幂, 天然对齐
    * 残差连接是 ResNet/Transformer 的标准结构 -> 更像真 AI
    * 残差本来就是为解决深层数值稳定发明的, 65536 层正需要它
"""

import ctypes
import hashlib
import time

import numpy as np
from numba import njit

W = 8
BLK = 64                       # 权重块 = 8x8 矩阵 = 64 个 double (2的幂)
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
TO_INT = 4503599627370496.0    # 2^52, 浮点->整数的确定性映射


# ---------------- 绑核 (修正 ctypes 类型) ----------------

def pin_and_boost(cpu=0):
    k32 = ctypes.windll.kernel32
    k32.GetCurrentThread.restype = ctypes.c_void_p
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k32.SetThreadAffinityMask.restype = ctypes.c_size_t
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    k32.SetPriorityClass.restype = ctypes.c_int
    k32.GetCurrentProcessorNumber.restype = ctypes.c_uint

    aff = k32.SetThreadAffinityMask(k32.GetCurrentThread(), ctypes.c_size_t(1 << cpu))
    pri = k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)   # HIGH
    time.sleep(0.05)
    return aff != 0, pri != 0, k32.GetCurrentProcessorNumber()


# ---------------- 频率标定 ----------------

@njit(cache=True)
def _lat_chain(n):
    """纯依赖链: imul(3周期) + xor(1周期) = 4 周期/次。"""
    x = np.uint64(1)
    for _ in range(n):
        x = ((x * np.uint64(6364136223846793005)) ^ np.uint64(0x9E3779B97F4A7C15)) & M64
    return x


def measure_ghz(reps=5):
    _lat_chain(1000)
    n = 30_000_000
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter(); _lat_chain(n); best = min(best, time.perf_counter() - t0)
    return n * 4.0 / best / 1e9


# ---------------- 阶段1: 廉价整数顺序填池 ----------------

@njit(cache=True)
def fill_pool(pool, s0, s1, mask):
    """
    xorshift128+ 风格的顺序链填池 + 全域随机回读。

    性质:
      * x[i] 依赖 x[i-1]         -> 无法并行填充
      * 起点由 (seed,nonce) 决定  -> 每 nonce 池子完全不同, CoW 免疫
      * 随机回读已写区域          -> 制造读写依赖 + DRAM 流量
      * 每个 double 仅 ~5 周期    -> 比 v2 的神经填池便宜 ~20 倍
    """
    n = pool.size
    x = s0
    y = s1
    for i in range(n):
        # xorshift128+ (顺序依赖)
        t = x
        t ^= (t << np.uint64(23)) & M64
        t ^= t >> np.uint64(17)
        t ^= y ^ (y >> np.uint64(26))
        x = y
        y = t
        v = (x + y) & M64

        # 全域随机回读 (池子零初始化, 读到未写区域也完全确定性)
        j = int(v & mask)
        v = (v ^ np.uint64(int(abs(pool[j]) * TO_INT))) & M64
        v = (v * np.uint64(0x9E3779B97F4A7C15)) & M64

        # 映射到 [-1,1), 避开 denormal
        pool[i] = (v >> np.uint64(11)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0


# ---------------- 阶段2: 神经主链 (残差 + 位掩码) ----------------

@njit(cache=True, fastmath=False)
def main_chain(pool, h, depth, blk_mask):
    """
    h <- act(W_t·h + h)   # 残差连接, 权重块 64 个 double
    权重块位置由上层激活决定 (指针追逐), 结果写回 (读改写依赖)。
    """
    a = h.copy()
    tmp = np.empty(W, dtype=np.float64)
    for _ in range(depth):
        # 由激活导出块索引 —— 位掩码, 不用取模
        acc = np.uint64(0xcbf29ce484222325)
        for j in range(W):
            acc = ((acc ^ np.uint64(int(abs(a[j]) * TO_INT))) * np.uint64(0x100000001b3)) & M64
        base = int(acc & blk_mask) * BLK

        for r in range(W):                       # W·h + h  (残差)
            s = a[r]
            off = base + r * W
            for c in range(W):
                s += pool[off + c] * a[c]
            tmp[r] = s

        s2 = 0.0                                 # 分支非线性 + 归一化
        for r in range(W):
            v = tmp[r]
            v = v / (1.0 + abs(v)) if v > 0.0 else 0.05 * v
            a[r] = v
            s2 += v * v
        inv = 1.0 / (np.sqrt(s2) + 1e-12)
        for r in range(W):
            a[r] *= inv

        pool[base] = pool[base] * 0.5 + a[0] * 0.5   # 写回
    return a


# ---------------- 顶层 ----------------

def narrownet(seed: bytes, nonce: int, pool_log2: int, depth: int):
    """pool_log2: 池子条目数 = 2^pool_log2 (必须 >= 6)"""
    n = 1 << pool_log2
    pool = np.zeros(n, dtype=np.float64)
    d = hashlib.sha256(seed + nonce.to_bytes(8, "big")).digest()
    s0 = np.uint64(int.from_bytes(d[0:8], "big") | 1)
    s1 = np.uint64(int.from_bytes(d[8:16], "big") | 1)
    fill_pool(pool, s0, s1, np.uint64(n - 1))

    # ⚠️ 必须给足 W 个 uint64。sha256 只有 32 字节 = 4 个 uint64,而 main_chain
    #    按 range(W)=8 读 a[j] —— numba njit 不做边界检查,会**静默读越界内存**,
    #    使同一 (seed,nonce) 在不同进程/机器上得出不同结果(PoW 直接失效)。
    #    用 sha512(64 字节 = 8 个 uint64)喂满。
    h = np.frombuffer(hashlib.sha512(d).digest(), dtype=np.uint64)[:W].copy()
    assert h.shape[0] == W, "初始激活必须恰好 W 个元素"
    h = (h >> np.uint64(11)).astype(np.float64) / 9007199254740992.0 * 2.0 - 1.0
    a = main_chain(pool, h, depth, np.uint64((n // BLK) - 1))
    return hashlib.sha256(a.tobytes()).digest(), pool


def main():
    aff, pri, cpu = pin_and_boost(0)
    print("=== NarrowNet v3 标定 ===")
    print(f"绑核: {'✅' if aff else '❌'}   高优先级: {'✅' if pri else '❌'}   当前运行于 CPU{cpu}\n")

    # 预热
    narrownet(b"warm", 0, 16, 500)
    ghz = measure_ghz()
    print(f"[实测频率] {ghz:.2f} GHz\n")

    # --- 填池成本 ---
    print("阶段1 填池成本 (v2 是神经填池, v3 换成整数顺序链):")
    for log2, mb in ((21, 16), (23, 64)):
        n = 1 << log2
        pool = np.zeros(n, dtype=np.float64)
        best = 1e18
        for _ in range(3):
            t0 = time.perf_counter()
            fill_pool(pool, np.uint64(12345), np.uint64(67891), np.uint64(n - 1))
            best = min(best, time.perf_counter() - t0)
        print(f"  {mb:>3} MB : {best*1000:7.1f} ms   ({best*1e9/n:.2f} ns/double, "
              f"{best*1e9/n*ghz:.1f} 周期)")

    # --- 主链成本 ---
    print("\n阶段2 主链成本 (残差 + 位掩码, 对比 v2 的偏置 + 取模):")
    print(f"  {'池子':>8}{'ns/步':>10}{'周期/步':>10}{'相对2MB':>10}")
    print("  " + "-" * 40)
    base = None
    res = {}
    for log2, mb in ((18, 2), (21, 16), (23, 64), (25, 256)):
        n = 1 << log2
        pool = np.zeros(n, dtype=np.float64)
        fill_pool(pool, np.uint64(1), np.uint64(2), np.uint64(n - 1))
        h = np.linspace(-0.5, 0.5, W)
        bm = np.uint64((n // BLK) - 1)
        main_chain(pool, h.copy(), 500, bm)
        depth = 200_000
        best = 1e18
        for _ in range(5):
            t0 = time.perf_counter(); main_chain(pool, h.copy(), depth, bm)
            best = min(best, time.perf_counter() - t0)
        ns = best * 1e9 / depth
        if base is None: base = ns
        res[mb] = ns
        print(f"  {mb:>6} MB{ns:>10.1f}{ns*ghz:>10.1f}{ns/base:>9.2f}x")
        del pool

    # --- 正确性 ---
    print("\n正确性验证:")
    d1, p1 = narrownet(b"seed", 7, 21, 20000)
    d2, _ = narrownet(b"seed", 7, 21, 20000)
    d3, p3 = narrownet(b"seed", 8, 21, 20000)
    same = int(np.count_nonzero(p1 == p3))
    diff = sum(bin(x ^ y).count("1") for x, y in zip(d1, d3))
    print(f"  确定性        : {d1 == d2}")
    print(f"  雪崩          : {diff}/256 (理想≈128)")
    print(f"  CoW免疫(池相同): {same}/{p1.size} ({same/p1.size*100:.4f}%)")
    print(f"  数值健康      : NaN={int(np.isnan(p1).sum())} Inf={int(np.isinf(p1).sum())} "
          f"denormal={int(np.count_nonzero((np.abs(p1)>0)&(np.abs(p1)<2.3e-308)))}")

    # --- 浏览器预算重算 ---
    print("\n=== 按实测重算浏览器参数 (目标: 手机 ~300ms) ===")
    ns64 = res[64]
    fill64_ms = (1 << 23) * (0.0 if 64 not in res else 1) * 0  # 用上面测的
    print(f"  假设手机WASM = 本机原生 x 3.5")
    for mb, log2 in ((16, 21), (64, 23)):
        n = 1 << log2
        # 填池已测, 主链按 res
        chain_ns = res[mb] * 3.5
        for target_ms in (300,):
            # 预留 30% 给填池
            depth = int(target_ms * 0.7 * 1e6 / chain_ns)
            print(f"  池子 {mb:>3} MB -> 深度 {depth:>9,} 步  (手机约 {target_ms} ms)")


if __name__ == "__main__":
    main()
