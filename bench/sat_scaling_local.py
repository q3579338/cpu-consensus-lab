#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAT-PoUW 的一票否决实验:CDCL 求解器在多核上是否线性扩展?

README 判据:「求解器多核线性扩展 -> 设计死亡」。若 N 核机器能跑满 N 个
独立求解器,大机器线性划算,"小机器优势"不成立。

--- 方法学:必须固定实例 ---
相变点随机 3SAT 的求解时间是**重尾分布**:同样规模换个种子能差几十倍。
若给不同并发数喂不同实例,测到的是实例难度差异,不是扩展性(本实验第一版
就栽在这里:标定 13.7s,实测 0.32s)。

所以这里让**每个并发度、每个 worker 都解同一个实例**——工作量逐位相同,
唯一变量是并发数。这与 contention.py 的口径一致(每线程持有同尺寸工作集)。

  单进程退化 = TN(并发时单个 worker 的耗时) / T1
  扩展效率   = T1 / TN     (1.0 = 完美线性,大机器全赚到)

对照组 SHA-256(工作集≈0)给出"这台机器无内存争用时的上限",
用来区分退化是 SAT 特有的,还是 SMT/功耗墙的固有限制。

7950X 关注点:1-8 单 CCD(32MB L3)| 16 跨双 CCD | 32 吃满 SMT
"""
import argparse, json, multiprocessing as mp, os, random, statistics, time

RATIO = 4.26                        # 相变点,最难
DEFAULT_COUNTS = [1, 2, 4, 8, 16, 32]


def gen_3sat(n_vars, seed):
    """确定性随机 3SAT(对应设计里"实例由区块头导出")。"""
    rng = random.Random(seed)
    cls = []
    for _ in range(int(n_vars * RATIO)):
        vs = rng.sample(range(1, n_vars + 1), 3)
        cls.append([v if rng.random() < 0.5 else -v for v in vs])
    return cls


def solve_worker(args):
    """一个独立矿工进程:解指定实例,返回纯求解耗时(不含生成)。"""
    n_vars, seed = args
    from pysat.solvers import Cadical153
    cls = gen_3sat(n_vars, seed)          # 生成不计时
    t0 = time.perf_counter()
    with Cadical153(bootstrap_with=cls) as s:
        s.solve()
    return time.perf_counter() - t0


def sha_worker(args):
    """对照组:SHA-256 链,工作集≈0,每 worker 工作量相同。"""
    seed, iters = args
    import hashlib
    t0 = time.perf_counter()
    h = b"seed" + str(seed).encode()
    for _ in range(iters):
        h = hashlib.sha256(h).digest()
    return time.perf_counter() - t0


def run_concurrent(fn, args_one, n_procs, warm=True):
    """起 n_procs 个进程,**每个都跑同一份 args_one**。
    返回 (墙钟, 各 worker 自报耗时列表)。"""
    with mp.Pool(processes=n_procs) as pool:
        if warm:
            pool.map(fn, [args_one])          # 预热 import/JIT
        t0 = time.perf_counter()
        times = pool.map(fn, [args_one] * n_procs)
        wall = time.perf_counter() - t0
    return wall, times


def pick_instance(target_lo, target_hi):
    """找一个 solo 耗时落在目标区间、且重复测量稳定的实例。"""
    n = 200
    while n < 1500:
        for seed in range(40):
            t = solve_worker((n, seed))
            if target_lo <= t <= target_hi:
                ts = [solve_worker((n, seed)) for _ in range(3)]
                spread = max(ts) / min(ts)
                print(f"  候选 n={n} seed={seed}: {min(ts):.2f}-{max(ts):.2f}s "
                      f"(离散 {spread:.2f}x)", flush=True)
                if spread < 1.25:                       # 稳定才用
                    return n, seed, statistics.median(ts)
        n = int(n * 1.3)
    raise SystemExit("没找到合适实例,调整 --target")


def bench(label, fn, args_one, counts, rounds, note_fn):
    print(f"\n=== {label} ===", flush=True)
    print(f"  {'并发':>5}{'单worker(s)':>13}{'退化':>8}{'扩展效率':>10}   备注", flush=True)
    t1 = None
    rows = []
    for n in counts:
        best = None
        for _ in range(rounds):
            wall, times = run_concurrent(fn, args_one, n)
            per = statistics.median(times)
            best = per if best is None else min(best, per)   # 取最小值滤噪
        if t1 is None:
            t1 = best
        degrade = best / t1
        eff = t1 / best
        flag = "线性" if eff > 0.85 else ("部分退化" if eff > 0.6 else "严重争用")
        print(f"  {n:>5}{best:>13.3f}{degrade:>7.2f}x{eff:>9.0%}   {note_fn(n)} {flag}", flush=True)
        rows.append({"threads": n, "per_worker_s": best, "degrade": degrade, "efficiency": eff})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-threads", type=int, default=32)
    ap.add_argument("--target", type=float, nargs=2, default=[2.0, 6.0],
                    help="单次求解目标耗时区间(秒)")
    a = ap.parse_args()

    counts = [c for c in DEFAULT_COUNTS if c <= a.max_threads]
    ncpu = mp.cpu_count()
    print("=== SAT-PoUW 一票否决实验:CDCL 多核扩展效率 ===", flush=True)
    print(f"机器 {ncpu} 逻辑核 | Cadical153 | 每点 {a.rounds} 轮取最小 | 相变比 {RATIO}", flush=True)
    print("方法:所有并发度共用同一实例,工作量恒定,唯一变量是并发数\n", flush=True)

    print("[挑选稳定实例]", flush=True)
    n_vars, seed, med = pick_instance(*a.target)
    print(f"  选定 n={n_vars} seed={seed} solo={med:.2f}s", flush=True)

    def note(n):
        return "单CCD内" if n <= 8 else ("跨双CCD" if n <= 16 else "吃满SMT")

    sat_rows = bench(f"SAT (Cadical, n={n_vars}, seed={seed})",
                     solve_worker, (n_vars, seed), counts, a.rounds, note)

    t0 = time.perf_counter(); sha_worker((0, 200000)); per = (time.perf_counter() - t0) / 200000
    iters = max(20000, int(med / per))
    sha_rows = bench(f"对照组 SHA-256 (工作集≈0, {iters} 轮)",
                     sha_worker, (0, iters), counts, a.rounds, note)

    out = {"cpu": "AMD Ryzen 9 7950X", "logical_cores": ncpu, "solver": "Cadical153",
           "n_vars": n_vars, "seed": seed, "solo_solve_s": med, "ratio": RATIO,
           "rounds": a.rounds, "method": "identical instance across all concurrency levels",
           "sat": sat_rows, "sha256_control": sha_rows}
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_7950x.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    def eff(rows, n):
        m = [r for r in rows if r["threads"] == n]
        return m[0]["efficiency"] if m else float("nan")

    print("\n=== 判据 ===", flush=True)
    print(f"  {'线程':>5}{'SAT效率':>10}{'SHA对照':>10}{'差值':>9}  <- 差值=SAT 独有的争用", flush=True)
    for n in counts:
        if n == 1:
            continue
        d = eff(sha_rows, n) - eff(sat_rows, n)
        print(f"  {n:>5}{eff(sat_rows,n):>9.0%}{eff(sha_rows,n):>10.0%}{d:>+9.0%}", flush=True)
    print("""
  SAT ≈ 对照 -> CDCL 线性扩展 -> **SAT-PoUW 小机器优势不成立(设计死亡)**
  SAT 明显更低 -> 存在 SAT 特有争用 -> 立论保住""", flush=True)
    print(f"结果 -> {os.path.normpath(dest)}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
