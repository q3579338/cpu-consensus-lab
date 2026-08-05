#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
写回消融:把 NarrowNet 主链的"写回"去掉(池子变只读),CPU/GPU 吞吐比退步多少?

背景
----
v3 主链每步末尾执行 `pool[base] = pool[base]*0.5 + a[0]*0.5`(写回)。
它在反 GPU 设计里的角色:让池子成为"会被写的内存"。只要存在哪怕一条写,
编译器/硬件就不能把池子的加载走只读路径(__ldg / L1 只读数据缓存 /
ld.global.nc),每次指针追逐都得按可写语义处理。这一环的价格从来没被
单独标过。

新的 v4-immutable 变体为了让"抽查式验证"可行(验证者随机抽若干步重算,
要求池子在整条链上不变),必须去掉写回。本脚本量化这一刀的代价:
去掉写回之后,CPU/GPU 吞吐比变成原来的几倍?

方法(2×2 消融:{有写回, 无写回} × {GPU, CPU})
----
  * GPU 只用严格 IEEE 语义(ld.dadd_rn/dmul_rn/ddiv_rn)。narrownet_cuda
    已判定只有它能与 CPU 逐位一致,FMA 变体是非法实现,没资格参赛。
  * 无写回 GPU kernel = narrownet_cuda._build_kernel_src() 的同一份生成
    源码删掉写回那一行(断言恰好删 1 行),其余逐字节相同 ——
    保证两个 kernel 之间的差异只有写回本身。
  * CPU 有写回 = 复用 narrownet_cuda.main_chain_nogil(即 src/narrownet_v3
    的 main_chain 以 nogil 重新 jit);CPU 无写回 = 从 v3 的 main_chain
    源码剥掉写回行后独立 jit(inspect.getsource + exec,不手抄一份,
    v3 改了这里自动跟着变或直接报错)。
  * 两对 (GPU, CPU) 各自逐位对拍,确认 4 个变体没有一个被改错。
  * 编译后检查 PTX:无写回 kernel 的 st.global 必须更少;若出现
    ld.global.nc 则说明编译器确实把池子放进了只读缓存路径(这正是
    写回要阻止的事)。

⚠️ CPU 侧两个变体都必须 njit(cache=False, fastmath=False, nogil=True)。
   numba 的磁盘缓存键**不含 nogil 标志**:narrownet_v3 的 main_chain 已以
   cache=True/nogil=False 写过磁盘缓存,这里若开 cache 会静默取回那份仍持
   GIL 的机器码,多线程比单线程还慢(narrownet_cuda 踩过:8 线程聚合
   7.31 Msteps/s < 单线程 9.17;关缓存后 28.95)。

用法
----
    python bench/writeback_ablation.py            # 完整(64MB/256MB 两档,必须空闲机器串行跑)
    python bench/writeback_ablation.py --quick    # 冒烟(2MB 小池、≤64 链、短步数,数字不作结论)

结果写入仓库根目录 writeback_ablation.json("mode" 字段区分 quick/full)。
"""
import inspect
import json
import math
import os
import statistics
import sys
import threading
import time

try:                                   # Windows 控制台默认 GBK,输出中文/符号会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

# 导入即执行 narrownet_cuda._ensure_nvvm()(CUDA 13 的 nvvm.dll 路径回退)
# —— 原样复用,不复制一份出来造成两处维护。
import narrownet_cuda as ncu                                # noqa: E402

import numpy as np                                          # noqa: E402
from numba import cuda, njit                                # noqa: E402
from numba.cuda import libdevice as ld                      # noqa: E402

import src.narrownet_v3 as v3                               # noqa: E402
from src.narrownet_v3 import main_chain, measure_ghz, pin_and_boost  # noqa: E402

W = ncu.W
BLK = ncu.BLK


# ============================================================================
# 变体构造 —— 全部从规范源码"删一行"派生,不手抄
# ============================================================================

def _kernel_src(writeback: bool) -> str:
    """取 narrownet_cuda 的规范 kernel 源码;无写回版只删写回一行。

    删除以字符串匹配定位("pools[base] = ADD"),并断言恰好命中 1 行:
    上游生成器若改了写回的写法,这里立刻炸,而不是静默测错东西。
    """
    src = ncu._build_kernel_src()
    if writeback:
        return src
    lines = src.splitlines()
    kept = [l for l in lines if "pools[base] = ADD" not in l]
    removed = len(lines) - len(kept)
    if removed != 1:
        raise RuntimeError(f"期望恰好删掉 1 行写回,实际删了 {removed} 行 —— "
                           "narrownet_cuda._build_kernel_src 可能改过,请同步本脚本")
    return "\n".join(kept)


def _compile_kernel(src):
    """严格 IEEE 语义编译(dadd_rn/dmul_rn/ddiv_rn)。不需要 FMA 变体:
    narrownet_cuda 已判定 FMA 与 CPU 不逐位一致 => 非法实现。"""
    ns = dict(cuda=cuda, np=np, math=math, M64=ncu.M64, TO_INT=ncu.TO_INT,
              FNV_OFF=ncu.FNV_OFF, FNV_PRIME=ncu.FNV_PRIME,
              ADD=ld.dadd_rn, MUL=ld.dmul_rn, DIV=ld.ddiv_rn)
    exec(src, ns)
    return cuda.jit(fastmath=False)(ns["chain_kernel"])


def _make_cpu_nowb():
    """CPU 无写回变体:从 v3 的 main_chain 源码剥掉写回行后独立 jit。

    必须 cache=False(见模块 docstring 的 nogil/缓存键坑)。改名为
    main_chain_nowb 只是为了调试栈可读,cache 已关,不存在键冲突问题。
    """
    src = inspect.getsource(main_chain.py_func)
    lines = src.splitlines()
    while lines and lines[0].lstrip().startswith("@"):      # 剥掉 @njit 装饰器行
        lines.pop(0)
    kept = [l for l in lines if not l.lstrip().startswith("pool[base] =")]
    removed = len(lines) - len(kept)
    if removed != 1:
        raise RuntimeError(f"期望恰好删掉 1 行写回,实际删了 {removed} 行 —— "
                           "narrownet_v3.main_chain 可能改过,请同步本脚本")
    src2 = "\n".join(kept).replace("def main_chain(", "def main_chain_nowb(", 1)
    ns = {"np": np, "W": v3.W, "BLK": v3.BLK, "M64": v3.M64, "TO_INT": v3.TO_INT}
    exec(src2, ns)
    return njit(cache=False, fastmath=False, nogil=True)(ns["main_chain_nowb"])


# ============================================================================
# 对拍与 PTX 侧证 —— 确认 4 个变体没有一个被改错
# ============================================================================

def correctness_pair(kern, cpu_fn, pool_log2=18, depth=3000):
    """同一个变体的 GPU/CPU 两侧必须逐位相同(池子同款 splitmix 填充)。"""
    n = 1 << pool_log2
    blk_mask = np.uint64((n // BLK) - 1)
    h0 = np.linspace(-0.5, 0.5, W)

    a_cpu = cpu_fn(ncu.fill_pools_host(n), h0.copy(), depth, blk_mask)

    d_pool = cuda.to_device(ncu.fill_pools_host(n))
    d_act = cuda.to_device(h0.reshape(1, W).copy())
    kern[1, 1](d_pool, d_act, depth, blk_mask, np.int64(n))
    cuda.synchronize()
    a_gpu = d_act.copy_to_host()[0]
    del d_pool, d_act

    bit_exact = bool(np.array_equal(a_gpu.view(np.uint64), a_cpu.view(np.uint64)))
    maxrel = float(np.max(np.abs(a_gpu - a_cpu) / (np.abs(a_cpu) + 1e-300)))
    return {"bit_exact": bit_exact, "max_rel_err": maxrel}


def ptx_stats(kern):
    """数 PTX 里的 store/只读加载。无写回版 st.global 必须更少;
    ld.global.nc 出现与否直接回答"写回是否真在阻止只读缓存"。"""
    try:
        asm = kern.inspect_asm()
        ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
        return {"st_global": ptx.count("st.global"),
                "ld_global_nc": ptx.count("ld.global.nc")}
    except Exception as e:                                   # numba 版本差异兜底
        return {"error": repr(e)}


# ============================================================================
# CPU 计时(参数化版 bench_cpu:两个 jit 变体共用同一套计时框架)
# ============================================================================

def bench_cpu(fn, pool_log2, n_threads, steps, reps=3):
    """返回 (聚合 Msteps/s, 每链 ns/步)。每线程独占池,与 GPU 每链独占池同构。"""
    pool_n = 1 << pool_log2
    blk_mask = np.uint64((pool_n // BLK) - 1)
    pools = [ncu.fill_pools_host(pool_n) for _ in range(n_threads)]
    hs = [np.linspace(-0.5, 0.5, W) + i * 1e-3 for i in range(n_threads)]

    for i in range(n_threads):                      # 预热 JIT
        fn(pools[i], hs[i].copy(), 200, blk_mask)

    best = 1e18
    for _ in range(reps):
        ths = [threading.Thread(target=fn,
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

    print("=== NarrowNet 写回消融:{有写回, 无写回} × {GPU, CPU} ===")
    print("模式: " + ("quick(冒烟,只验证端到端能出数,数字不作结论)" if quick else "full"))
    gpu = cuda.get_current_device()
    free_b, total_b = cuda.current_context().get_memory_info()
    ram = psutil.virtual_memory()
    phys = psutil.cpu_count(logical=False) or 16
    print(f"GPU : {gpu.name.decode()}  显存 {total_b/2**30:.1f} GB(空闲 {free_b/2**30:.1f} GB)")
    print(f"CPU : {ncu.ECON['cpu_name']}  {phys}C  内存可用 {ram.available/2**30:.1f} GB")
    busy = [p.info["name"] for p in psutil.process_iter(["name"])
            if p.info["name"] == "python.exe"]
    if len(busy) > 1:
        print(f"⚠️  另有 {len(busy)-1} 个 python 进程在跑,数字可能被污染(完整跑必须独占机器)")

    pin_and_boost(0)
    ghz = measure_ghz()
    print(f"实测频率: {ghz:.2f} GHz\n")

    print("编译 4 个变体(GPU×2 严格 IEEE,CPU×2 nogil/cache=False)...", flush=True)
    kern_wb = _compile_kernel(_kernel_src(writeback=True))
    kern_nowb = _compile_kernel(_kernel_src(writeback=False))
    cpu_wb = ncu.main_chain_nogil        # v3 main_chain 的 nogil 重 jit(cache=False)
    cpu_nowb = _make_cpu_nowb()

    variants = [("writeback", "有写回", kern_wb, cpu_wb),
                ("readonly", "无写回", kern_nowb, cpu_nowb)]

    # ---------------- 对拍 ----------------
    depth_c = 500 if quick else 3000
    print(f"\n=== 对拍(pool 2^18,depth {depth_c};每个变体的 GPU 必须与自己的 CPU 逐位相同)===")
    cr = {}
    for key, zh, kern, cfn in variants:
        cr[key] = correctness_pair(kern, cfn, pool_log2=18, depth=depth_c)
        tag = ("✅ 逐位相同" if cr[key]["bit_exact"]
               else f"❌ 不同(max_rel {cr[key]['max_rel_err']:.2e})")
        print(f"  {zh}({key}) {tag}")
    if not all(r["bit_exact"] for r in cr.values()):
        print("  ⚠️ 有变体两侧不一致 —— 变体改错了,下面的吞吐数字不可作数")

    # ---------------- PTX 侧证 ----------------
    print("\n=== PTX 侧证(无写回版 st.global 必须更少)===")
    ptx = {key: ptx_stats(kern) for key, _zh, kern, _cfn in variants}
    for key, _zh, _k, _c in variants:
        s = ptx[key]
        if "error" in s:
            print(f"  {key:<9} inspect_asm 失败: {s['error']}")
        else:
            print(f"  {key:<9} st.global × {s['st_global']:<3}  ld.global.nc × {s['ld_global_nc']}")
    if ("error" not in ptx["writeback"] and "error" not in ptx["readonly"]):
        if ptx["readonly"]["st_global"] >= ptx["writeback"]["st_global"]:
            print("  ⚠️ 无写回版的 st.global 没有变少 —— 写回可能没删干净,勿信下面的数字")
        if ptx["readonly"]["ld_global_nc"] == 0:
            print("  (注:无写回版也没出现 ld.global.nc —— 编译器没走只读缓存路径,"
                  "则本消融测到的主要是 store 本身与读改写依赖的代价)")

    # ---------------- 吞吐 ----------------
    ladder = [(18, 2)] if quick else [(23, 64), (25, 256)]
    steps_cpu = 10_000 if quick else 100_000
    target_s = 0.5 if quick else 3.0
    chain_cap = 64 if quick else ncu.GPU_CHAIN_CAP
    budget = free_b * ncu.VRAM_BUDGET
    ram_budget = ram.available * 0.55

    print(f"\n=== 吞吐(显存预算 {budget/2**30:.1f} GB,内存预算 {ram_budget/2**30:.1f} GB)===")
    print(f"{'池子':>7}{'变体':>6}{'GPU链数':>8}{'排布':>6}{'GPU ns/步':>11}{'GPU吞吐':>10}"
          f"{'CPU线程':>8}{'CPU吞吐':>10}{'CPU/GPU':>9}")
    print("-" * 84)

    rows = []
    for log2, mb in ladder:
        pool_b = (1 << log2) * 8
        n_chains = min(chain_cap, int(budget // pool_b))
        if n_chains < 1:
            print(f"{mb:>5} MB   显存放不下一条链,跳过")
            continue
        n_threads = max(1, min(4 if quick else 16, phys, int(ram_budget // pool_b)))

        row = {"pool_mb": mb, "pool_log2": log2, "cpu_threads": n_threads}
        for key, zh, kern, cfn in variants:
            g_msteps, g_ns, tpb, watt, n_used = ncu.bench_gpu(
                kern, n_chains, log2, target_s=target_s)
            c_msteps, c_ns = bench_cpu(cfn, log2, n_threads, steps_cpu)
            ratio = c_msteps / g_msteps
            row[key] = {"gpu_chains": n_used, "tpb": tpb,
                        "gpu_ns_per_step": g_ns, "gpu_msteps": g_msteps,
                        "gpu_peak_watt": watt,
                        "cpu_msteps": c_msteps, "cpu_ns_per_step": c_ns,
                        "ratio_cpu_over_gpu": ratio}
            print(f"{mb:>5} MB{zh:>6}{n_used:>8}{tpb:>6}{g_ns:>11.0f}{g_msteps:>9.2f}M"
                  f"{n_threads:>8}{c_msteps:>9.2f}M{ratio:>8.2f}x", flush=True)
        row["ratio_change_readonly_over_writeback"] = (
            row["readonly"]["ratio_cpu_over_gpu"] / row["writeback"]["ratio_cpu_over_gpu"])
        rows.append(row)

    # ---------------- 结论 ----------------
    print("\n=== 结论 ===")
    if not rows:
        print("  没有可用数据")
        return
    for r in rows:
        wb = r["writeback"]["ratio_cpu_over_gpu"]
        ro = r["readonly"]["ratio_cpu_over_gpu"]
        chg = r["ratio_change_readonly_over_writeback"]
        verdict = ("CPU 相对退步" if chg < 1 else "CPU 反而相对受益(反直觉,建议复查)")
        print(f"  {r['pool_mb']:>5} MB : CPU/GPU 比 {wb:.2f}x -> {ro:.2f}x,"
              f"即变为原来的 {chg:.2f}x({verdict} {max(chg, 1/chg):.2f} 倍)")
    med = statistics.median(r["ratio_change_readonly_over_writeback"] for r in rows)
    print(f"\n  去掉写回(池子只读)让 CPU/GPU 吞吐比变为原来的 {med:.2f}x(中位数)"
          + (";quick 模式数字仅供冒烟,不作结论" if quick else ""))

    dest = os.path.join(_ROOT, "writeback_ablation.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"mode": "quick" if quick else "full",
                   "gpu": gpu.name.decode(), "cpu": ncu.ECON["cpu_name"], "ghz": ghz,
                   "correctness": cr, "ptx": ptx, "rows": rows,
                   "median_ratio_change_readonly_over_writeback": med},
                  f, indent=2, ensure_ascii=False)
    print(f"  -> {dest}")


if __name__ == "__main__":
    main()
