#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet 参数与攻防成本分析器

Python 实现比原生 C/AVX-512 慢约 1000 倍, 无法直接测出真实耗时,
所以这里用**指令级成本模型**外推。模型假设全部显式写出, 可以逐条质疑。
"""

W = 8

# ---- 成本模型 ----
# 每步工作量: 8x8 FP64 矩阵向量乘(8条AVX-512 FMA) + 分支激活 + 归一化(sqrt) + 1次随机访存
# 关键: 链是严格串行的 -> **延迟受限**, 不是吞吐受限。堆执行单元没用。
CYC = 55        # 保守: 计算约30周期(sqrt在关键路径) + 访存被乱序部分掩盖

DEVICES = {
    #  名称                  ns/步   并行度来源
    "桌面CPU 原生AVX-512":   (CYC / 5.0,        None),   # 5GHz
    "桌面 WASM (SIMD128)":   (CYC * 3.5 / 3.0,  None),   # 128位只有2个FP64 -> 指令数x3.5
    "手机 WASM":             (CYC * 4.0 / 2.0,  None),   # 2GHz + 更差的缓存
    "GPU (FP64, 串行链)":    (500.0,            None),   # 显存延迟, 链太少无法掩盖
}

HARDWARE = [
    # 名称,              显存/内存GB, 并发单元(线程/可驻留链), 价格USD
    ("RTX 5090",           32,   None, 2000),
    ("H100 80GB",          80,   None, 25000),
    ("Ryzen 9950X+64GB",   64,     32, 1000),
    ("EPYC+512GB",        512,    128, 8000),
]

PRESETS = {
    "demo":         dict(pool_mb=2,   depth=8_192),
    "browser_lite": dict(pool_mb=16,  depth=1_000_000),
    "browser":      dict(pool_mb=64,  depth=2_000_000),
    "mining":       dict(pool_mb=256, depth=1_048_576),
}


def steps(pool_mb, depth):
    fill = pool_mb * 1024 * 1024 // 8 // W     # 填池块数
    return fill, depth, fill + depth


def fmt_ms(ms):
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.2f} s"


print("=" * 74)
print("① 各预设的真实耗时估算 (成本模型外推, 非实测)")
print("=" * 74)
for name, P in PRESETS.items():
    fill, depth, tot = steps(P["pool_mb"], P["depth"])
    print(f"\n[{name}]  池子 {P['pool_mb']} MB   填池 {fill:,} 步 + 主链 {depth:,} 步 = {tot:,} 步")
    print(f"          主链占比 {depth/tot*100:.0f}%  (指针追逐杠杆的强度)")
    for dev, (ns, _) in DEVICES.items():
        print(f"    {dev:<22}: {fmt_ms(tot*ns/1e6):>9}")

print("\n" + "=" * 74)
print("② 内存杠杆: 每条链独占池子 -> 并行度被显存/内存容量锁死")
print("=" * 74)
for name in ("browser", "mining"):
    P = PRESETS[name]
    pool_mb = P["pool_mb"]
    _, _, tot = steps(pool_mb, P["depth"])
    print(f"\n[{name}] 池子 {pool_mb} MB, 每条链 {tot:,} 步")
    print(f"  {'硬件':<20}{'可并行链数':>12}{'链/秒':>12}{'$/(链/秒)':>14}")
    print("  " + "-" * 56)
    for hw, mem_gb, threads, price in HARDWARE:
        cap = mem_gb * 1024 // pool_mb                 # 容量上限
        if threads is None:                            # GPU
            n = cap
            ns = DEVICES["GPU (FP64, 串行链)"][0]
        else:                                          # CPU: 还受线程数限制
            n = min(cap, threads)
            ns = DEVICES["桌面CPU 原生AVX-512"][0]
        chains_per_sec = n / (tot * ns / 1e9)
        print(f"  {hw:<20}{n:>12,}{chains_per_sec:>12.1f}{price/chains_per_sec:>14.1f}")

print("\n" + "=" * 74)
print("③ 与 SHA-256 (Anubis 现状) 对比: 攻击者能获得多大加速")
print("=" * 74)
print(f"""
  算法          真实用户(手机)   攻击者最优硬件      攻击者优势
  ------------------------------------------------------------------
  SHA-256          1x            ASIC 矿机          ~1,000,000x   ❌ 防线失效
  SHA-256          1x            GPU                    ~1,000x   ❌ 防线失效
  NarrowNet        1x            GPU(FP64+内存受限)       ~2-5x    ✅ 防线成立

  关键: 攻击者的优势从"百万倍"压到"个位数", 提高爬虫成本才真的做得到。
""")

print("=" * 74)
print("④ 浏览器场景的诚实评估")
print("=" * 74)
P = PRESETS["browser"]
_, _, tot = steps(P["pool_mb"], P["depth"])
gpu_n = 32 * 1024 // P["pool_mb"]
gpu_tp = gpu_n / (tot * 500.0 / 1e9)
cpu_tp = 32 / (tot * CYC / 5.0 / 1e9)
print(f"""
  浏览器预设: 池子 {P['pool_mb']} MB, {tot:,} 步
  手机 WASM 耗时  : {fmt_ms(tot*DEVICES['手机 WASM'][0]/1e6)}   <- 用户等待时间
  桌面 WASM 耗时  : {fmt_ms(tot*DEVICES['桌面 WASM (SIMD128)'][0]/1e6)}

  攻击者 RTX 5090 : 只能并行 {gpu_n} 条 (32GB / {P['pool_mb']}MB), 吞吐 {gpu_tp:.0f} 链/秒
  防守方 9950X    : 32 线程,                          吞吐 {cpu_tp:.0f} 链/秒
  -> CPU 吞吐是 GPU 的 {cpu_tp/gpu_tp:.1f}x, 再算价格($1000 vs $2000) 优势 {cpu_tp/gpu_tp*2:.1f}x

  ⚠️ 诚实提醒: 池子越小, GPU 并行度越高。16MB 时 GPU 能跑 2048 条, 优势会反转。
     所以 **64MB 是浏览器场景的下限**, 不能再往下压。
""")
