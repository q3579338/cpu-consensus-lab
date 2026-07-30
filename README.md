# NarrowNet — A CPU-Favoring, GPU-Resistant Proof-of-Work

*[中文说明见下方](#中文说明) · [完整攻击面分析](docs/security-analysis.zh.md)*

A proof-of-work function whose computation **is** a neural-network forward pass — but shaped
so that every dimension works against GPUs: **width 8, depth in the millions, FP64, and a
per-nonce mutable weight pool that caps GPU parallelism at `VRAM ÷ pool_size`.**

> **This repo reports measured numbers, including the ones that contradict its own earlier design claims.**
> A cost model that was off by 15× is documented in full, along with what it invalidated.

---

## The idea in one table

Real AI workloads are exactly what GPUs are built for. So invert every dimension:

| Dimension | Conventional AI (GPU's home turf) | NarrowNet (CPU's home turf) |
|---|---|---|
| Width | 4096+ | **8** (one AVX-512 FP64 vector) |
| Depth | tens of layers | **10⁵–10⁶ layers** |
| Batch | thousands | **1** |
| Precision | FP16 / BF16 | **FP64, strict IEEE** |
| Weights | resident in VRAM, reused | **fetched per-layer from a mutable pool, address chosen by the previous activation** |

It is still a genuine neural network — matrix-vector products, a nonlinearity, residual
connections, and normalization — just in a shape no trained model would ever use.

## Defense levers (and which ones turned out to be fake)

| Lever | Blocks | Strength |
|---|---|---|
| **Per-chain exclusive memory pool** | Caps GPU parallelism at `VRAM ÷ pool`, independent of core count | ⭐⭐⭐⭐⭐ |
| **Per-nonce pool (CoW immunity)** | Blocks the copy-on-write bypass of the above | ⭐⭐⭐⭐⭐ |
| **FP64 division** (8 per step) | Consumer GPUs run FP64 at 1/64 rate; division multiplies that penalty | ⭐⭐⭐⭐ |
| **FP64 strict IEEE** | Silicon-level handicap on consumer GPUs | ⭐⭐⭐⭐ |
| **Serial dependency chain** | No intra-chain parallelism | ⭐⭐⭐⭐ |
| **Pointer chasing** | Address depends on previous activation — no prefetch, no coalescing | ⭐⭐⭐⭐ |
| ~~Narrow width wastes warps~~ | ❌ **Fails** under one-thread-per-nonce | — |
| ~~Kernel launch overhead~~ | ❌ **Fails** under a persistent megakernel | — |

The last two were claimed in early design notes and are **retracted** — a rational attacker
runs one independent chain per GPU thread, which defeats both.

## Measured results

Intel Core Ultra 5 125H (Meteor Lake, **no AVX-512**), pinned to a P-core, 4.25 GHz measured,
Numba/LLVM JIT native code, minimum of 5 runs.

**The memory lever is real:**

| Pool | ns/step | cycles/step | vs 2 MB |
|---|---|---|---|
| 2 MB | 65.4 | 278 | 1.00× |
| 16 MB | 130.2 | 553 | 1.99× |
| 64 MB | 212.3 | 902 | 3.25× |
| 256 MB | 245.5 | 1043 | **3.75×** |

**Correctness:**

```
Determinism (same seed+nonce)     : True
Avalanche (1-bit nonce change)    : 132/256 bits  (ideal ≈128)
CoW immunity (pool overlap)       : 0 / 2,097,152 entries  (0.0000%)
Numerical health                  : NaN=0  Inf=0  denormal=0
```

## ⚠️ The headline finding: pool size decides everything

Using **measured** CPU cost (not the earlier 15×-wrong model), against an estimated GPU at
~420 ns/step (memory-latency-bound at low occupancy):

| Pool | GPU parallel chains | CPU/GPU throughput | CPU/GPU per dollar | Verdict |
|---|---|---|---|---|
| 16 MB | 2,048 | 0.09× | 0.18× | ❌ GPU wins big |
| 64 MB | 512 | 0.22× | 0.45× | ❌ GPU wins |
| 256 MB | 128 | 0.78× | 1.56× | ~ tie |
| **1 GB** | **32** | **2.93×** | **5.85×** | ✅ CPU wins |
| **2 GB** | **16** | **5.43×** | **10.86×** | ✅ CPU wins decisively |

**A pool of ≥1 GB is required for a decisive CPU advantage.**

This independently reproduces the rationale behind Monero RandomX's 2 GB dataset — that
parameter is not arbitrary, it is the threshold where the memory lever actually bites.

### What this means for browser use (anti-bot / anti-AI-scraper)

Browsers cannot allocate 1 GB, so **NarrowNet does not give CPUs an advantage in a browser.**
But "CPU beats GPU" was never the right goal for bot defense. The right goal is *the attacker
must not get an order-of-magnitude shortcut*:

| Algorithm | Max attacker speedup |
|---|---|
| SHA-256 (what [Anubis](https://github.com/TecharoHQ/anubis) uses today) | **~1,000,000×** (ASIC) |
| **NarrowNet @ 64 MB** | **~4.5×** (GPU) |

Dropping the attacker's edge from six orders of magnitude to less than one is a ~200,000×
improvement in defense quality — even though the CPU does not "win". **That is the only claim
this project makes for the browser case.**

## Status — what is NOT verified

1. **No real GPU measurement.** The 420 ns/step GPU figure is an *estimate*. This project just
   demonstrated that estimates can be wrong by 15×, so treat every CPU-vs-GPU number here as
   provisional until a CUDA implementation exists. **This is the single largest gap.**
2. No hand-written C/AVX-512 implementation (test machine lacks AVX-512).
3. No WASM build or real mobile measurement.
4. No formal argument that no algebraic shortcut exists.
5. No third-party cryptographic review.

## Usage

```bash
pip install numpy numba
python src/narrownet_v3.py      # main implementation + self-calibration
python bench/bench_v2.py        # step-cost benchmark
```

`src/narrownet_v3.py` is the current implementation. `v1`/`v2` are kept to document the
CoW vulnerability and its fix.

## Layout

```
src/narrownet_v3.py   Current implementation (power-of-2 pool, residual, cheap integer fill)
src/narrownet_v2.py   Introduced per-nonce pools — fixes the copy-on-write bypass
src/narrownet_v1.py   Original design (CoW-vulnerable, kept for reference)
bench/                Native calibration + attack-economics model
docs/                 Full attack-surface analysis (Chinese)
vdf/                  Wesolowski VDF — a separate, complementary primitive (see below)
```

## Also here: a Wesolowski VDF

`vdf/` contains an unrelated but complementary primitive: repeated squaring in an RSA group
with O(1) verification. Where NarrowNet is a probabilistic PoW (verification means recomputing),
a VDF *proves elapsed sequential time* and verifies in milliseconds — the right tool for
time-locks, sealed-bid auctions, and randomness beacons.

Measured (T = 10⁶ sequential squarings, 1024-bit modulus):

| | v1 (naive) | v2 (checkpointed, parallel) |
|---|---|---|
| eval (strictly serial) | 2101 ms | 2101 ms |
| prove | 11827 ms | **3823 ms** (3.1× faster) |
| verify | **4.83 ms** | 4.83 ms |

`eval` cannot be parallelized by anyone — that is the point. `prove` *can* be, and v2 exploits it.

## Prior work

| Project | Relation |
|---|---|
| [RandomX](https://github.com/tevador/RandomX) | Same goal (CPU-favoring PoW), different method — random x86 code generation. Its 2 GB dataset threshold is independently confirmed here. |
| [Argon2](https://github.com/P-H-C/phc-winner-argon2) | Sequential memory-hard filling; the pool-fill phase borrows this idea. |
| [Coin.AI](https://arxiv.org/pdf/1903.09800) | PoUW via DNN *training*; NarrowNet uses inference *shape* without useful output. |
| [Anubis](https://github.com/TecharoHQ/anubis) | The target application — currently SHA-256-based, hence GPU/ASIC-bypassable. |

No existing implementation of "neural-inference-shaped anti-GPU PoW" was found in two rounds of
search. Individual ingredients (memory-hardness, pointer chasing, serial chains, AI+PoW) are all
prior art; **the combination appears to be unoccupied.** That is not a guarantee of novelty.

## License

MIT

---

<a name="中文说明"></a>

# 中文说明

一个**计算过程本身就是神经网络前向传播**的工作量证明算法，但把每一个维度都调到 GPU 的反面：
**宽度 8、深度百万级、FP64、以及每个 nonce 独占的可变权重池——后者把 GPU 的并行度锁死为 `显存 ÷ 池子大小`。**

> **本仓库如实报告实测数据，包括那些推翻了自己早期设计宣称的数据。**
> 一个偏差 15 倍的成本模型被完整记录，连同它导致的错误结论。

## 核心思路

真实 AI 计算恰恰是 GPU 的主场。所以把每个维度反过来：

| 维度 | 常规 AI（GPU 主场） | NarrowNet（CPU 主场） |
|---|---|---|
| 宽度 | 4096+ | **8**（一个 AVX-512 FP64 向量） |
| 深度 | 几十层 | **10⁵–10⁶ 层** |
| batch | 几千 | **1** |
| 精度 | FP16 / BF16 | **FP64 严格 IEEE** |
| 权重 | 常驻显存反复复用 | **每层从可变池中取，位置由上一层激活决定** |

它仍然是货真价实的神经网络——矩阵向量乘、非线性激活、残差连接、归一化——只是形状是任何训练出来的模型都不会用的。

## 防线（以及哪些是纸老虎）

| 防线 | 挡什么 | 强度 |
|---|---|---|
| **每链独占内存池** | GPU 并行度锁死为 `显存 ÷ 池子`，与核心数无关 | ⭐⭐⭐⭐⭐ |
| **每 nonce 独立池（CoW 免疫）** | 堵住写时复制对上一条的绕过 | ⭐⭐⭐⭐⭐ |
| **FP64 除法**（每步 8 次） | 消费级 GPU 的 FP64 是 1/64 速率，除法惩罚在此之上再叠加 | ⭐⭐⭐⭐ |
| **FP64 严格 IEEE** | 消费级 GPU 的硅片级阉割 | ⭐⭐⭐⭐ |
| **串行依赖链** | 单链内部无法并行 | ⭐⭐⭐⭐ |
| **指针追逐** | 地址依赖上一层激活，无法预取、无法合并访存 | ⭐⭐⭐⭐ |
| ~~窄宽度填不满 warp~~ | ❌ **在「一线程一 nonce」下失效** | — |
| ~~kernel launch 开销~~ | ❌ **在 megakernel 下失效** | — |

最后两条在早期设计笔记里宣称过，现已**撤回**——理性的攻击者会让每个 GPU 线程跑一条独立的链，这两条同时失效。

## ⚠️ 最重要的结论：池子大小决定成败

用**实测**的 CPU 成本（而非那个错了 15 倍的模型）重算：

| 池子 | GPU 可并行 | CPU/GPU 吞吐 | 单位成本比 | 判定 |
|---|---|---|---|---|
| 16 MB | 2,048 | 0.09× | 0.18× | ❌ GPU 完胜 |
| 64 MB | 512 | 0.22× | 0.45× | ❌ GPU 胜 |
| 256 MB | 128 | 0.78× | 1.56× | ~ 打平 |
| **1 GB** | **32** | **2.93×** | **5.85×** | ✅ CPU 胜 |
| **2 GB** | **16** | **5.43×** | **10.86×** | ✅ CPU 完胜 |

**必须 ≥1 GB 才有决定性的 CPU 优势。** 这独立复现了门罗币 RandomX 采用 2 GB 数据集的理由——那个参数不是随便取的，而是内存杠杆真正生效的门槛。

### 对浏览器场景（反爬虫）意味着什么

浏览器分配不了 1 GB，所以 **NarrowNet 在浏览器里拿不到 CPU 对 GPU 的优势**。但"CPU 赢 GPU"本来就不是反爬的正确目标，正确目标是**攻击者不能获得数量级的加速捷径**：

| 算法 | 攻击者最大加速 |
|---|---|
| SHA-256（Anubis 现在用的） | **~1,000,000×**（ASIC） |
| **NarrowNet @ 64 MB** | **~4.5×**（GPU） |

把攻击者的优势从 6 个数量级压到不足 1 个——防御质量提升约 20 万倍，**即使 CPU 并没有"赢"**。这是本项目对浏览器场景唯一的宣称，不夸大。

## 尚未验证（最大的坑）

1. **没有真实 GPU 实测。** 那个 420 ns/步是**估算**。本项目刚刚亲手证明了估算能错 15 倍，所以在有 CUDA 实现之前，所有 CPU-vs-GPU 数字都应视为暂定。**这是当前最大的缺口。**
2. 没有手写 C/AVX-512 实现（测试机无 AVX-512）
3. 没有 WASM 构建和真机测试
4. 没有"不存在代数捷径"的形式化论证
5. 没有第三方密码学审计

## 运行

```bash
pip install numpy numba
python src/narrownet_v3.py      # 主实现 + 自标定
python bench/bench_v2.py        # 每步成本基准测试
```

## 标定方法学（踩过的坑）

1. **混合架构必须绑核** —— Intel Core Ultra 是 P 核 + E 核，Windows 会迁移线程，E 核慢 2–3 倍。未绑核时同一函数测出 183 和 527 ns 两个值。`SetThreadAffinityMask` 经 ctypes 调用**必须声明 argtypes/restype**，否则静默失败。
2. **取最小值不取平均** —— 测延迟的标准做法。
3. **频率要实测** —— 用已知延迟的依赖链反推；矛盾的读数是线程被迁移的信号。
4. **`fastmath=False` 是强制的** —— PoW 要求逐位确定性，编译器重排浮点会改变舍入。

## 关于原创性

两轮检索没有找到"以神经网络推理形态构建的反 GPU PoW"的现成实现。但单个要素（内存硬、指针追逐、串行链、AI+PoW）**都是已有工作**，新的只是这个组合。**这不构成原创性保证。**

## 许可

MIT
