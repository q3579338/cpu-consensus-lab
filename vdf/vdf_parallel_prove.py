#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VDF v2 —— 检查点 + 多核并行证明

相对 v1 的改进:
  v1: prove 用朴素的 pow(x, q, N), 代价 O(T), 比 eval 还慢。
  v2: eval 过程中顺路存 k 个检查点, prove 拆成 k 个独立幂运算并行跑。

核心恒等式 (s = T/k):
    q  = ⌊2^T / l⌋ = Σⱼ qⱼ · 2^(js)      (把 q 按 2^s 进制拆开)
    π  = x^q = ∏ⱼ (x^(2^(js)))^(qⱼ) = ∏ⱼ Cⱼ^(qⱼ)
其中 Cⱼ = x^(2^(js)) 就是 eval 时存下的第 j 个检查点。
这 k 项互不依赖 -> 可以全核并行。

体现的不对称正是我们要的:
    eval  : 严格串行, 谁也没法加速  (延迟的来源, GPU 无解)
    prove : 可并行,   核越多越快    (只是证明, 不构成延迟)
    verify: O(1),     毫秒级
"""

import hashlib
import secrets
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor


# ---------------- 素性 / hash-to-prime ----------------

def _is_probable_prime(n: int, rounds: int = 32) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _i2b(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8 or 1, "big")


def hash_to_prime(*chunks: bytes, bits: int = 256) -> int:
    """确定性地把输入映射到一个素数 l, verifier 能复现。"""
    counter = 0
    while True:
        h = hashlib.sha256()
        for c in chunks:
            h.update(len(c).to_bytes(8, "big"))
            h.update(c)
        h.update(counter.to_bytes(8, "big"))
        acc, i = b"", 0
        while len(acc) * 8 < bits:
            acc += hashlib.sha256(h.digest() + i.to_bytes(4, "big")).digest()
            i += 1
        cand = int.from_bytes(acc[: bits // 8], "big") | (1 << (bits - 1)) | 1
        if _is_probable_prime(cand):
            return cand
        counter += 1


def gen_modulus(bits: int = 1024):
    def rand_prime(b):
        while True:
            c = secrets.randbits(b) | (1 << (b - 1)) | 1
            if _is_probable_prime(c):
                return c
    p, q = rand_prime(bits // 2), rand_prime(bits // 2)
    return p * q, (p, q)


# ---------------- eval: 串行平方 + 存检查点 ----------------

def eval_with_checkpoints(x: int, T: int, N: int, k: int):
    """
    做 T 次串行平方求 y = x^(2^T) mod N，
    同时每 s=T/k 步存一个检查点 Cⱼ = x^(2^(js))。
    存检查点几乎不增加开销（只是偶尔拷一个整数）。
    """
    s = T // k
    ckpts = []
    y = x % N
    for i in range(T):
        if i % s == 0 and len(ckpts) < k:
            ckpts.append(y)          # 此刻 y == x^(2^i)
        y = y * y % N
    return y, ckpts, s


# ---------------- prove: 并行 ----------------

def _seg_pow(args):
    """单个检查点的幂运算，交给子进程。"""
    C, qj, N = args
    return pow(C, qj, N)


def prove_parallel(x: int, y: int, T: int, N: int, ckpts, s: int, workers: int):
    """
    π = ∏ⱼ Cⱼ^(qⱼ)，k 项并行。
    """
    l = hash_to_prime(_i2b(x), _i2b(y), _i2b(N), _i2b(T))
    q = (1 << T) // l                      # 大整数除法，O(T) bit 运算，很快
    mask = (1 << s) - 1

    tasks = []
    for j, C in enumerate(ckpts):
        qj = (q >> (j * s)) & mask         # q 的第 j 段（2^s 进制的第 j 位）
        if qj:
            tasks.append((C, qj, N))

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_seg_pow, tasks))
    else:
        parts = [_seg_pow(t) for t in tasks]

    pi = 1
    for p in parts:
        pi = pi * p % N
    return pi, l


def prove_naive(x: int, y: int, T: int, N: int):
    """v1 的朴素做法，用来对照。"""
    l = hash_to_prime(_i2b(x), _i2b(y), _i2b(N), _i2b(T))
    return pow(x, (1 << T) // l, N), l


# ---------------- verify: O(1) ----------------

def verify(x: int, y: int, T: int, N: int, pi: int, l: int) -> bool:
    if hash_to_prime(_i2b(x), _i2b(y), _i2b(N), _i2b(T)) != l:
        return False
    r = pow(2, T, l)
    return (pow(pi, l, N) * pow(x, r, N)) % N == y % N


# ---------------- 演示 ----------------

def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    cores = os.cpu_count() or 4
    k = min(cores, 16)                     # 检查点数 = 并行度
    T = (T // k) * k                       # 让 T 能被 k 整除

    print(f"[*] CPU 核心: {cores}  |  检查点/并行度 k = {k}")
    print(f"[*] 生成 1024-bit 模数 ...")
    N, (p, _) = gen_modulus(1024)
    x = (int.from_bytes(hashlib.sha256(secrets.token_bytes(32)).digest(), "big") % N) | 1

    print(f"[*] T = {T:,} 次串行平方\n")

    # --- eval (串行, 不可加速) ---
    t0 = time.perf_counter()
    y, ckpts, s = eval_with_checkpoints(x, T, N, k)
    t1 = time.perf_counter()
    ev = (t1 - t0) * 1000
    print(f"eval  (严格串行)      : {ev:9.1f} ms   [{ev*1000/T:.3f} µs/平方, 存了 {len(ckpts)} 个检查点]")

    # --- prove 朴素 ---
    t2 = time.perf_counter()
    pi_n, l_n = prove_naive(x, y, T, N)
    t3 = time.perf_counter()
    pn = (t3 - t2) * 1000
    print(f"prove (v1 朴素单核)   : {pn:9.1f} ms")

    # --- prove 并行 ---
    t4 = time.perf_counter()
    pi, l = prove_parallel(x, y, T, N, ckpts, s, workers=k)
    t5 = time.perf_counter()
    pp = (t5 - t4) * 1000
    print(f"prove (v2 检查点并行) : {pp:9.1f} ms   <- 提速 {pn/pp:.1f}x")

    # --- verify ---
    t6 = time.perf_counter()
    ok = verify(x, y, T, N, pi, l)
    t7 = time.perf_counter()
    vf = (t7 - t6) * 1000
    print(f"verify                : {vf:9.2f} ms   -> {'✔ 通过' if ok else '✗ 失败'}")

    print(f"\n[=] 两种 prove 结果一致: {pi == pi_n}")
    print(f"[=] eval / verify = {ev/vf:,.0f}x   (延迟价值)")
    print(f"[=] eval / prove  = {ev/pp:.2f}x    (>1 表示证明已不再是瓶颈)")

    # --- 安全性反证 ---
    print(f"[=] 伪造 y: {'✗ 被拒(正确)' if not verify(x,(y+1)%N,T,N,pi,l) else '✔ 通过(不该!)'}")
    print(f"[=] 伪造 π: {'✗ 被拒(正确)' if not verify(x,y,T,N,(pi+1)%N,l) else '✔ 通过(不该!)'}")


if __name__ == "__main__":
    main()
