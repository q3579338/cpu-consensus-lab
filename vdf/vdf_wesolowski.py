#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VDF (Verifiable Delay Function) —— 基于 RSA 群上重复平方 + Wesolowski 证明

核心保证:
  y = x^(2^T) mod N
  必须一步一步平方 T 次才能算出 y —— 这是"内在串行(inherently sequential)"。
  没有已知算法能并行这条链, 也无法用多核/GPU 批处理来加速"单条链"。
  谁堆再多核心, 单条 VDF 的墙钟时间都由单核平方速度决定。

  验证却是 O(1): 拿到 (y, π) 后只需两三次幂运算, 毫秒级完成。

安全前提:
  - N = p*q 的分解必须无人知晓(否则可用欧拉定理走捷径: 2^T mod φ(N) 直接跳)。
    生产环境用"无人知道分解"的模数, 比如 RSA-2048 挑战数, 或多方安全生成后销毁 p,q。
  - 本 demo 为可复现会临时生成 p,q 并"假装"丢弃, 仅演示用。
"""

import hashlib
import secrets
import time
import sys


# ---------- 工具: 从字节确定性派生素数 (hash-to-prime) ----------

def _is_probable_prime(n: int, rounds: int = 32) -> bool:
    """Miller-Rabin 素性测试。"""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def hash_to_prime(*chunks: bytes, bits: int = 256) -> int:
    """
    把任意输入确定性地映射成一个素数 l。
    做法: 用计数器不断哈希, 直到落到一个素数上。verifier 能复现同一个 l。
    """
    counter = 0
    while True:
        h = hashlib.sha256()
        for c in chunks:
            h.update(len(c).to_bytes(8, "big"))
            h.update(c)
        h.update(counter.to_bytes(8, "big"))
        # 拉伸到目标位数
        acc = b""
        i = 0
        while len(acc) * 8 < bits:
            acc += hashlib.sha256(h.digest() + i.to_bytes(4, "big")).digest()
            i += 1
        cand = int.from_bytes(acc[: bits // 8], "big")
        cand |= (1 << (bits - 1)) | 1  # 置高位 + 置奇数
        if _is_probable_prime(cand):
            return cand
        counter += 1


def _i2b(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8 or 1, "big")


# ---------- 群参数生成 ----------

def gen_modulus(bits: int = 1024):
    """
    生成 N = p*q。返回 (N, (p, q))。
    真实部署: 生成后必须销毁 p,q, 且最好用可信多方或公开挑战数。
    """
    def rand_prime(b):
        while True:
            cand = secrets.randbits(b) | (1 << (b - 1)) | 1
            if _is_probable_prime(cand):
                return cand
    p = rand_prime(bits // 2)
    q = rand_prime(bits // 2)
    return p * q, (p, q)


# ---------- VDF 三件套 ----------

def eval_vdf(x: int, T: int, N: int) -> int:
    """
    评估: y = x^(2^T) mod N, 老老实实做 T 次串行平方。
    这是唯一(已知)的算法路径 —— 无法并行, 无法跳步。
    """
    y = x % N
    for _ in range(T):
        y = y * y % N
    return y


def prove(x: int, y: int, T: int, N: int):
    """
    Wesolowski 证明。生成一个常数大小的 π。
      l = hash_to_prime(x, y)
      q = floor(2^T / l),  π = x^q mod N
    """
    l = hash_to_prime(_i2b(x), _i2b(y), _i2b(N), _i2b(T))
    q = (1 << T) // l
    pi = pow(x, q, N)   # 这一步用快速幂即可, 因为指数 q 已知(它不受"串行"约束)
    return pi, l


def verify(x: int, y: int, T: int, N: int, pi: int, l: int) -> bool:
    """
    验证: 检查 π^l · x^r ≡ y (mod N), 其中 r = 2^T mod l。
    重算一遍 l 防止 prover 作弊换 l。全程只有几次幂运算 -> 毫秒级。
    """
    l_check = hash_to_prime(_i2b(x), _i2b(y), _i2b(N), _i2b(T))
    if l_check != l:
        return False
    r = pow(2, T, l)
    lhs = (pow(pi, l, N) * pow(x, r, N)) % N
    return lhs == y % N


# ---------- 演示 ----------

def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

    print(f"[*] 生成 1024-bit RSA 模数 N (真实部署需销毁 p,q)...")
    N, (p, q) = gen_modulus(1024)

    # 从随机种子确定性得到起点 x
    seed = secrets.token_bytes(32)
    x = int.from_bytes(hashlib.sha256(seed).digest(), "big") % N
    x |= 1

    print(f"[*] T = {T:,} 次串行平方")
    print(f"[*] 开始评估 (这段时间无法用任何并行手段缩短)...")
    t0 = time.perf_counter()
    y = eval_vdf(x, T, N)
    t1 = time.perf_counter()
    eval_ms = (t1 - t0) * 1000
    print(f"    eval 用时: {eval_ms:.1f} ms  ({eval_ms*1000/T:.3f} µs/平方)")

    print(f"[*] 生成证明 π...")
    t2 = time.perf_counter()
    pi, l = prove(x, y, T, N)
    t3 = time.perf_counter()
    print(f"    prove 用时: {(t3-t2)*1000:.1f} ms")

    print(f"[*] 验证...")
    t4 = time.perf_counter()
    ok = verify(x, y, T, N, pi, l)
    t5 = time.perf_counter()
    verify_ms = (t5 - t4) * 1000
    print(f"    verify 用时: {verify_ms:.2f} ms  ->  {'✔ 通过' if ok else '✗ 失败'}")

    print(f"\n[=] eval / verify 时间比 ≈ {eval_ms/verify_ms:,.0f}x  (这就是'延迟'的价值)")

    # 反向验证: 伪造的 y 必须被拒
    bad = verify(x, (y + 1) % N, T, N, pi, l)
    print(f"[=] 伪造 y 的验证结果: {'✔ 通过(不该!)' if bad else '✗ 被拒(正确)'}")

    # 证明"捷径": 知道 p,q 就能秒算 (说明为何必须销毁分解)
    phi = (p - 1) * (q - 1)
    e = pow(2, T, phi)          # 2^T mod φ(N), 把 T 次平方压成一次幂
    y_shortcut = pow(x, e, N)
    print(f"[=] 用私钥(p,q)走捷径算出的 y 与串行结果一致: {y_shortcut == y}  <- 所以分解必须无人知晓")


if __name__ == "__main__":
    main()
