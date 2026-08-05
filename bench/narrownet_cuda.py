#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NarrowNet 的一票否决实验:真实 CUDA 实测,不再估算。

README 里每一个 CPU-vs-GPU 数字,GPU 那一半都是**估算的**(420 ns/步)。
本项目已经证明这类估算能错 15 倍——上一次纠错把"64MB 池 CPU 赢 2.8x"
翻转成了"GPU 赢 4.5x"。这个脚本把估算换成同一台机器上的实测。

被检验的三个假设(NarrowNet 的全部赌注):
  H1  强制 FP64      —— 消费级 GPU 的 FP64 是 FP32 的 1/64
  H2  每 nonce 独占池 —— GPU 并行度被 VRAM ÷ pool 卡死,与核心数无关
  H3  严格 IEEE      —— GPU 默认把 a+b*c 合并成 FMA(单次舍入),结果与 CPU
                        逐位不同 => 解无效。合法的 GPU 矿工必须用 dadd_rn/
                        dmul_rn 关掉合并,而这本身有性能代价。H3 的代价从来
                        没被量化过,这里第一次量。

方法学(刻意偏向 GPU,否则测出来的是"我不会写 kernel"而不是"GPU 不适合"):
  * 8 个激活值用 8 个**标量**而非 local array —— 保证落在寄存器里
  * 64 次乘加全部展开成直线代码,无循环开销
  * tpb ∈ {1, 32} 两种排布都跑,取更快的那个
  * 池子填充用并行 splitmix64(填充成本不是本实验的对象,v3 已单独标定过)
  * CPU 侧直接复用 src/narrownet_v3.py 的同一份 main_chain,禁止另写一份

用法:
    python bench/narrownet_cuda.py            # 完整
    python bench/narrownet_cuda.py --quick    # 冒烟(小池子、短步数)
"""
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import warnings


def _ensure_nvvm():
    """CUDA 13 把 nvvm64_40_0.dll 挪进了 nvvm/bin/x64/,numba 还按老路径找。
    若当前 CUDA_PATH 下没有老布局,就退回到装了老布局的那个版本。"""
    root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    cur = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    if cur and os.path.isfile(os.path.join(cur, "nvvm", "bin", "nvvm64_40_0.dll")):
        os.environ["CUDA_HOME"] = cur
        return cur
    for d in sorted(glob.glob(os.path.join(root, "v*")), reverse=True):
        if os.path.isfile(os.path.join(d, "nvvm", "bin", "nvvm64_40_0.dll")):
            os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = d
            return d
    return cur


_NVVM_HOME = _ensure_nvvm()

try:                                   # Windows 控制台默认 GBK,输出中文/符号会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                          # noqa: E402
from numba import cuda, njit                                # noqa: E402
from numba.cuda import libdevice as ld                      # noqa: E402
from numba.core.errors import NumbaPerformanceWarning       # noqa: E402

# 低占用率正是本实验要测的东西,不需要 numba 反复提醒
warnings.simplefilter("ignore", NumbaPerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.narrownet_v3 import main_chain, measure_ghz, pin_and_boost  # noqa: E402

# ---- 与 narrownet_v3 完全一致的常量 ----
W = 8
BLK = 64
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
TO_INT = 4503599627370496.0
FNV_OFF = np.uint64(0xCBF29CE484222325)
FNV_PRIME = np.uint64(0x100000001B3)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)

# ---- 经济性假设(不是实测,改这里即可重算)----
ECON = {
    "cpu_name": "Ryzen 9 7950X",
    "gpu_name": "RTX 4090",
    "cpu_usd": 550.0,      # 散片行价
    "gpu_usd": 1600.0,     # 行价
    "cpu_watt": 230.0,     # PPT 标称,非实测(Windows 上没有便捷的 RAPL)
}

VRAM_BUDGET = 0.72        # 只用空闲显存的这个比例,给桌面/驱动留活路
GPU_CHAIN_CAP = 8192      # 上限,防止小池子时链数爆炸导致跑不完
TDR_CHUNK_S = 1.0         # 单次 kernel 目标时长(本机 TdrDelay=60s,仍保守)


# ============================================================================
# CUDA kernel —— 单份源码,两套算子语义(合并 FMA / 严格 IEEE)
# ============================================================================

def _build_kernel_src():
    """生成主链 kernel 源码。ADD/MUL/DIV 在 exec 命名空间里被替换成两种语义。"""
    L = []
    A = L.append
    A("def chain_kernel(pools, acts, depth, blk_mask, pool_n):")
    A("    i = cuda.grid(1)")
    A("    if i >= acts.shape[0]:")
    A("        return")
    A("    base_off = np.int64(i) * pool_n")
    for r in range(W):
        A(f"    a{r} = acts[i, {r}]")
    A("    for _ in range(depth):")
    # --- FNV-1a over the 8 activations (指针追逐的来源) ---
    A("        acc = FNV_OFF")
    for j in range(W):
        A(f"        acc = ((acc ^ np.uint64(int(abs(a{j}) * TO_INT))) * FNV_PRIME) & M64")
    A("        base = base_off + np.int64(acc & blk_mask) * 64")
    # --- 残差矩阵向量乘: t_r = a_r + sum_c pool[base+r*8+c] * a_c ---
    #     累加顺序必须与 CPU 版逐字一致(左结合),否则舍入不同
    for r in range(W):
        A(f"        t{r} = a{r}")
        for c in range(W):
            A(f"        t{r} = ADD(t{r}, MUL(pools[base + {r * W + c}], a{c}))")
    # --- 分支非线性 + L2 归一化 ---
    A("        s2 = 0.0")
    for r in range(W):
        A(f"        v = t{r}")
        A("        if v > 0.0:")
        A("            v = DIV(v, ADD(1.0, abs(v)))")
        A("        else:")
        A("            v = MUL(0.05, v)")
        A(f"        a{r} = v")
        A("        s2 = ADD(s2, MUL(v, v))")
    A("        inv = DIV(1.0, ADD(math.sqrt(s2), 1e-12))")
    for r in range(W):
        A(f"        a{r} = MUL(a{r}, inv)")
    # --- 写回(读改写依赖,阻断只读缓存共享)---
    A("        pools[base] = ADD(MUL(pools[base], 0.5), MUL(a0, 0.5))")
    for r in range(W):
        A(f"    acts[i, {r}] = a{r}")
    return "\n".join(L)


@cuda.jit(device=True, inline=True)
def _add_fast(x, y):
    return x + y


@cuda.jit(device=True, inline=True)
def _mul_fast(x, y):
    return x * y


@cuda.jit(device=True, inline=True)
def _div_fast(x, y):
    return x / y


def _compile_kernels():
    src = _build_kernel_src()
    base = dict(cuda=cuda, np=np, math=math, M64=M64, TO_INT=TO_INT,
                FNV_OFF=FNV_OFF, FNV_PRIME=FNV_PRIME)
    out = {}
    for name, (add, mul, div) in {
        "fma":   (_add_fast, _mul_fast, _div_fast),          # 允许编译器合成 FMA
        "exact": (ld.dadd_rn, ld.dmul_rn, ld.ddiv_rn),       # 严格 IEEE,禁止合并
    }.items():
        ns = dict(base, ADD=add, MUL=mul, DIV=div)
        exec(src, ns)
        out[name] = cuda.jit(fastmath=False)(ns["chain_kernel"])
    return out


@cuda.jit(fastmath=False)
def fill_kernel(pools, seed):
    """splitmix64(index) -> [-1,1)。并行填充:填池成本不是本实验的对象。"""
    i = cuda.grid(1)
    if i >= pools.size:
        return
    z = (np.uint64(i) + seed) & M64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M64
    z = z ^ (z >> np.uint64(31))
    pools[i] = (z >> np.uint64(11)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0


def fill_pools_host(n):
    """CPU 侧同款填充,保证两边池子逐位相同(仅用于正确性对拍)。"""
    idx = np.arange(n, dtype=np.uint64)
    z = idx
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0


# ============================================================================
# 采样 GPU 功耗
# ============================================================================

class PowerSampler:
    def __init__(self):
        self.vals, self._stop, self._th = [], threading.Event(), None

    def _run(self):
        while not self._stop.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4)
                self.vals.append(float(o.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(timeout=2)

    @property
    def peak(self):
        return max(self.vals) if self.vals else float("nan")


# ============================================================================
# 正确性对拍 —— H3 就在这里被判定
# ============================================================================

def correctness(kernels, pool_log2=18, depth=3000):
    n = 1 << pool_log2
    blk_mask = np.uint64((n // BLK) - 1)
    h0 = np.linspace(-0.5, 0.5, W)

    pool_cpu = fill_pools_host(n)
    a_cpu = main_chain(pool_cpu.copy(), h0.copy(), depth, blk_mask)

    res = {}
    for name, kern in kernels.items():
        d_pool = cuda.to_device(fill_pools_host(n))
        d_act = cuda.to_device(h0.reshape(1, W).copy())
        kern[1, 1](d_pool, d_act, depth, blk_mask, np.int64(n))
        cuda.synchronize()
        a_gpu = d_act.copy_to_host()[0]
        bit_exact = bool(np.array_equal(a_gpu.view(np.uint64), a_cpu.view(np.uint64)))
        maxrel = float(np.max(np.abs(a_gpu - a_cpu) / (np.abs(a_cpu) + 1e-300)))
        res[name] = {"bit_exact": bit_exact, "max_rel_err": maxrel}
        del d_pool, d_act
    return res


# ============================================================================
# GPU / CPU 计时
# ============================================================================

def _launch(kern, d_pools, d_acts, steps, blk_mask, pool_n, tpb):
    n = d_acts.shape[0]
    blocks = (n + tpb - 1) // tpb
    kern[blocks, tpb](d_pools, d_acts, steps, blk_mask, pool_n)


def bench_gpu(kern, n_chains, pool_log2, target_s=3.0):
    """返回 (最好排布的 Msteps/s, ns/步/链, tpb, 峰值功耗)。"""
    pool_n = 1 << pool_log2
    blk_mask = np.uint64((pool_n // BLK) - 1)

    d_pools = None
    while d_pools is None:                       # 显存要不到就减链数,别直接崩
        try:
            d_pools = cuda.device_array(n_chains * pool_n, dtype=np.float64)
        except Exception:
            cuda.current_context().deallocations.clear()
            n_chains //= 2
            if n_chains < 1:
                raise
            print(f"    (显存不足,链数降到 {n_chains})", flush=True)
    tpb_fill = 256
    fill_kernel[(d_pools.size + tpb_fill - 1) // tpb_fill, tpb_fill](
        d_pools, np.uint64(12345))
    cuda.synchronize()

    rng = np.random.default_rng(7)
    acts0 = rng.random((n_chains, W)) * 2 - 1

    best = None
    for tpb in (1, 32):
        if tpb > n_chains:
            continue
        d_acts = cuda.to_device(acts0.copy())
        # 预热 + 标定单次 kernel 该跑多少步(控制在 TDR_CHUNK_S 内)
        _launch(kern, d_pools, d_acts, 64, blk_mask, np.int64(pool_n), tpb)
        cuda.synchronize()
        t0 = time.perf_counter()
        _launch(kern, d_pools, d_acts, 512, blk_mask, np.int64(pool_n), tpb)
        cuda.synchronize()
        per_step = (time.perf_counter() - t0) / 512
        chunk = max(64, min(200_000, int(TDR_CHUNK_S / max(per_step, 1e-12))))
        total = max(chunk, int(target_s / max(per_step, 1e-12)))

        with PowerSampler() as ps:
            cuda.synchronize()
            t0 = time.perf_counter()
            done = 0
            while done < total:
                s = min(chunk, total - done)
                _launch(kern, d_pools, d_acts, s, blk_mask, np.int64(pool_n), tpb)
                done += s
            cuda.synchronize()
            dt = time.perf_counter() - t0
        msteps = total * n_chains / dt / 1e6
        ns_chain = dt * 1e9 / total
        if best is None or msteps > best[0]:
            best = (msteps, ns_chain, tpb, ps.peak, n_chains)
        del d_acts
    del d_pools
    cuda.current_context().deallocations.clear()
    return best


# CPU 侧:同一份 main_chain,加 nogil 才能真正多线程。
#
# ⚠️ cache 必须为 False。numba 的磁盘缓存键**不包含 nogil 标志**,
#    narrownet_v3 里的 main_chain(nogil=False)已经写过缓存,这里若开 cache
#    会静默取回那份仍持 GIL 的机器码 —— 实测 8 线程聚合 7.31 Msteps/s,
#    比单线程的 9.17 还低;关掉缓存后是 28.95。整个 CPU 侧会被这一个字废掉。
main_chain_nogil = njit(cache=False, fastmath=False, nogil=True)(main_chain.py_func)


def bench_cpu(pool_log2, n_threads, steps, reps=3):
    """返回 (聚合 Msteps/s, 每链 ns/步)。每线程持有自己的池,与 GPU 语义一致。"""
    pool_n = 1 << pool_log2
    blk_mask = np.uint64((pool_n // BLK) - 1)
    pools = [fill_pools_host(pool_n) for _ in range(n_threads)]
    hs = [np.linspace(-0.5, 0.5, W) + i * 1e-3 for i in range(n_threads)]

    for i in range(n_threads):                      # 预热 JIT
        main_chain_nogil(pools[i], hs[i].copy(), 200, blk_mask)

    best = 1e18
    for _ in range(reps):
        ths = [threading.Thread(target=main_chain_nogil,
                                args=(pools[i], hs[i].copy(), steps, blk_mask))
               for i in range(n_threads)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        best = min(best, time.perf_counter() - t0)
    del pools
    return steps * n_threads / best / 1e6, best * 1e9 / steps


# ============================================================================
# main
# ============================================================================

def main():
    quick = "--quick" in sys.argv
    import psutil

    print("=== NarrowNet:真实 CUDA 实测 vs 同机 CPU ===\n")
    gpu = cuda.get_current_device()
    free_b, total_b = cuda.current_context().get_memory_info()
    ram = psutil.virtual_memory()
    print(f"GPU : {gpu.name.decode()}  CC {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
          f"  显存 {total_b/2**30:.1f} GB(空闲 {free_b/2**30:.1f} GB)")
    print(f"CPU : {ECON['cpu_name']}  {psutil.cpu_count(logical=False)}C/"
          f"{psutil.cpu_count()}T  内存 {ram.total/2**30:.0f} GB(可用 {ram.available/2**30:.1f} GB)")
    busy = [p.info["name"] for p in psutil.process_iter(["name", "cpu_percent"])
            if p.info["name"] == "python.exe"]
    if len(busy) > 1:
        print(f"⚠️  另有 {len(busy)-1} 个 python 进程在跑,CPU 侧数字可能被污染")

    pin_and_boost(0)
    ghz = measure_ghz()
    print(f"实测频率: {ghz:.2f} GHz\n")

    print("编译 kernel(两套算子语义)...", flush=True)
    kernels = _compile_kernels()

    # ---------------- H3:严格 IEEE 能不能在 GPU 上复现 ----------------
    print("\n=== H3 正确性对拍(GPU 结果必须与 CPU 逐位相同,否则解无效)===")
    cr = correctness(kernels, pool_log2=18, depth=3000)
    for name, r in cr.items():
        tag = "✅ 逐位相同" if r["bit_exact"] else f"❌ 不同(最大相对误差 {r['max_rel_err']:.2e})"
        print(f"  {name:<6} {tag}")
    legal = [k for k, v in cr.items() if v["bit_exact"]]
    print(f"  -> 合法的 GPU 实现只能用: {legal or '(无)'}")

    # ---------------- H1/H2:吞吐对比 ----------------
    ladder = [(21, 16), (23, 64)] if quick else [(18, 2), (21, 16), (23, 64), (25, 256), (27, 1024)]
    steps_cpu = 20_000 if quick else 100_000
    budget = free_b * VRAM_BUDGET
    ram_budget = ram.available * 0.55

    print(f"\n=== H1/H2 吞吐(显存预算 {budget/2**30:.1f} GB,内存预算 {ram_budget/2**30:.1f} GB)===")
    print("  (CPU 单线程 ns/步 可直接对比 README 里的笔记本数字;若它≈多线程值则绑核没串台)")
    print(f"{'池子':>7}{'GPU链数':>8}{'排布':>6}{'GPU ns/步':>11}{'GPU吞吐':>10}"
          f"{'CPU线程':>8}{'单线程ns/步':>12}{'CPU吞吐':>10}{'CPU/GPU':>9}")
    print("-" * 88)

    rows = []
    for log2, mb in ladder:
        pool_b = (1 << log2) * 8
        n_chains = min(GPU_CHAIN_CAP, int(budget // pool_b))
        n_threads = max(1, min(psutil.cpu_count(logical=False), int(ram_budget // pool_b)))
        if n_chains < 1:
            print(f"{mb:>5} MB   显存放不下一条链,跳过")
            continue

        row = {"pool_mb": mb, "gpu_chains": n_chains, "cpu_threads": n_threads}
        for name, kern in kernels.items():
            g_msteps, g_ns, tpb, watt, n_used = bench_gpu(
                kern, n_chains, log2, target_s=1.0 if quick else 3.0)
            row[f"gpu_{name}"] = {"msteps": g_msteps, "ns_per_step": g_ns, "tpb": tpb,
                                  "peak_watt": watt, "chains": n_used}
            row["gpu_chains"] = min(row["gpu_chains"], n_used)

        c1_msteps, c1_ns = bench_cpu(log2, 1, steps_cpu)
        c_msteps, c_ns = bench_cpu(log2, n_threads, steps_cpu)
        cores = psutil.cpu_count(logical=False)
        row["cpu"] = {"msteps": c_msteps, "ns_per_step": c_ns,
                      "single_msteps": c1_msteps, "single_ns_per_step": c1_ns,
                      # 内存不够时按当前并发下的每链延迟线性外推到满核。
                      # 忽略了更多线程带来的额外争用 => 这是对 CPU 有利的**上界**。
                      "proj_full_cores_msteps": cores * 1000.0 / c_ns,
                      "ram_capped": n_threads < cores,
                      # 两边各自的内存天花板:池子越大,谁都被卡,不只是 GPU
                      "hw_ceiling_chains": int(ram.total // pool_b)}

        # 只跟**合法**变体比:非法变体算得再快也换不到有效解
        key = f"gpu_{legal[0]}" if legal else "gpu_exact"
        g = row[key]
        row["ratio_cpu_over_gpu"] = c_msteps / g["msteps"]
        row["compared_against"] = key
        print(f"{mb:>5} MB{n_chains:>8}{g['tpb']:>6}{g['ns_per_step']:>11.0f}"
              f"{g['msteps']:>9.2f}M{n_threads:>8}{c1_ns:>12.1f}{c_msteps:>9.2f}M"
              f"{row['ratio_cpu_over_gpu']:>8.2f}x")
        rows.append(row)

    # ---------------- 汇总 ----------------
    print("\n=== 判读 ===")
    if not rows:
        print("  没有可用数据")
        return
    for r in rows:
        g = r[r["compared_against"]]
        per_usd = ((r["cpu"]["msteps"] / ECON["cpu_usd"]) /
                   (g["msteps"] / ECON["gpu_usd"]))
        w = g["peak_watt"]
        per_w = ((r["cpu"]["msteps"] / ECON["cpu_watt"]) /
                 (g["msteps"] / w)) if w == w else float("nan")
        r["cpu_per_usd_over_gpu"] = per_usd
        r["cpu_per_watt_over_gpu"] = per_w
        verdict = "✅ CPU 赢" if r["ratio_cpu_over_gpu"] > 1 else "❌ GPU 赢"
        print(f"  {r['pool_mb']:>5} MB : 吞吐 {r['ratio_cpu_over_gpu']:>6.2f}x   "
              f"每美元 {per_usd:>6.2f}x   每瓦 {per_w:>6.2f}x   {verdict}")
        if r["cpu"]["ram_capped"]:
            proj = r["cpu"]["proj_full_cores_msteps"] / g["msteps"]
            print(f"            ↑ CPU 只开了 {r['cpu_threads']} 线程(内存不够)。"
                  f"满核上界外推 = {proj:.2f}x;本机内存天花板 "
                  f"{r['cpu']['hw_ceiling_chains']} 链 vs 显存 {r['gpu_chains']} 链")

    fma = [r for r in rows if "gpu_fma" in r and "gpu_exact" in r]
    if fma:
        cost = statistics.median(r["gpu_fma"]["msteps"] / r["gpu_exact"]["msteps"]
                                 for r in fma)
        print(f"\n  H3 的代价:禁止 FMA 合并让 GPU 慢了 {cost:.2f}x(中位数)")

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "narrownet_gpu_vs_cpu.json")
    with open(dest, "w") as f:
        json.dump({"gpu": gpu.name.decode(), "cpu": ECON["cpu_name"], "ghz": ghz,
                   "econ": ECON, "correctness": cr, "rows": rows}, f, indent=2)
    print(f"\n  -> {dest}")


if __name__ == "__main__":
    main()
