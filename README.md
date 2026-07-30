# CPU Consensus Lab

*[中文说明见下方](#中文说明)*

Three designs attacking one question: **can consensus work be made to favor CPUs and small
machines over GPUs — and if so, at what cost?**

This is a lab, not a product. Each design is carried far enough to identify the single experiment
that would kill it, and those experiments are named explicitly. One of them has already fired: a
cost model here was **wrong by 15×**, and what it invalidated is documented rather than quietly
edited out.

---

## The three designs

| | **NarrowNet** | **PoRT** | **SAT-PoUW** |
|---|---|---|---|
| **Scarce resource** | Memory capacity | Network position | Solver quality |
| **Anti-GPU mechanism** | Manufactured (FP64, serial chain, per-nonce pool) | Network latency dominates critical path | **Inherent to the problem** |
| **Verification cost** | Full recomputation | ~640 ms | ✅ **O(n) — check the assignment** |
| **Is the work useful?** | No | No | ✅ **Yes** |
| **Measured?** | ✅ Calibrated | ❌ No | ❌ No |
| **What would kill it** | Pool < 1 GB → GPU wins | Per-IP quota signal too thin | Solvers scale linearly across cores |
| **Central risk** | Needs ≥1 GB per instance | IP wholesale (/24) attack | **Secret solver = mining advantage** |

The trajectory across the three is the interesting part: NarrowNet and PoRT go to considerable
length to *manufacture* GPU-hostility. SAT solving simply **is** that shape — and two decades of
failed GPU SAT research is itself the security property.

---

### NarrowNet — a neural network shaped against GPUs

`src/narrownet_v3.py` · [full attack-surface analysis](docs/security-analysis.zh.md) (Chinese)

A proof-of-work whose computation *is* a neural-network forward pass, with every dimension
inverted from what GPUs are built for:

| Dimension | Conventional AI | NarrowNet |
|---|---|---|
| Width | 4096+ | **8** (one AVX-512 FP64 vector) |
| Depth | tens of layers | **10⁵–10⁶** |
| Batch | thousands | **1** |
| Precision | FP16 / BF16 | **FP64, strict IEEE** |
| Weights | resident in VRAM, reused | **per-layer fetch from a mutable pool, address set by the previous activation** |

The load-bearing lever is the **per-nonce mutable pool**, which caps GPU parallelism at
`VRAM ÷ pool_size` regardless of core count. A copy-on-write bypass of exactly this was found and
fixed in v2 (measured: 0 / 2,097,152 shared entries between nonces).

**Measured** (Core Ultra 5 125H, no AVX-512, P-core pinned, 4.25 GHz, min of 5 runs):

| Pool | ns/step | cycles/step | vs 2 MB |
|---|---|---|---|
| 2 MB | 65.4 | 278 | 1.00× |
| 16 MB | 130.2 | 553 | 1.99× |
| 64 MB | 212.3 | 902 | 3.25× |
| 256 MB | 245.5 | 1043 | **3.75×** |

**The finding that reversed the design's own conclusion:**

| Pool | GPU parallel chains | CPU/GPU throughput | per dollar | |
|---|---|---|---|---|
| 64 MB | 512 | 0.22× | 0.45× | ❌ GPU wins |
| 256 MB | 128 | 0.78× | 1.56× | ~ tie |
| **1 GB** | **32** | **2.93×** | **5.85×** | ✅ |
| **2 GB** | **16** | **5.43×** | **10.86×** | ✅ |

An earlier version of this README claimed a 2.8× CPU advantage at 64 MB. That was built on the
15×-wrong cost model; the real figure is GPU winning by 4.5×. **A pool of ≥1 GB is required** —
independently reproducing the rationale behind Monero RandomX's 2 GB dataset.

---

### PoRT — Proof of Round Trips

[protocol spec](docs/port-protocol.zh.md) (Chinese)

Moves the scarce resource off FLOPS entirely. Each attempt requires **K = 64 sequential network
round-trips**, where the next peer to query is determined by the *signature* of the previous
response — so nothing can be prefetched, pipelined, or parallelized.

```
for i in 1..64:
    state ← memory_step(state, 4 MB scratchpad)   # ~10 ms, cache-contended
    peer  ← active_set[H(state) mod N]            # unpredictable until now
    resp  ← peer.request(state)                   # ~20 ms RTT, unavoidable
    state ← H(state ‖ resp.sig)                   # feeds the next peer choice
```

Latency becomes ~67% of the critical path: a GPU that reduced local compute to zero would gain
only a third, while 64 serial round-trips leave every streaming multiprocessor idle. The optimal
unit becomes a well-connected **1-core / 1 GB VPS** — you scale by adding cheap globally
distributed instances, not cores. Bandwidth is ~27 GB/month.

Its per-instance quota mechanism has a **known serious gap** (§8.1 of the spec): with a large
active set, rate-limit signal per responder is too thin to detect a multiplexer.

---

### SAT-PoUW — where the work is actually useful

[design doc](docs/sat-pouw.zh.md) (Chinese)

CDCL SAT solvers are the textbook irregular workload — branch-dense, pointer-chasing through
watched literals, strongly serial through conflict-driven learning. No manufacturing required.

It also has the best verification asymmetry available anywhere in this repo: solving is
NP-complete, **checking an assignment is O(n)**.

Two tiers keep utility from touching consensus safety:

- **Consensus** — synthetic 3SAT instances derived deterministically from the block header.
  Tunable difficulty, unpredictable, infinite supply, no external issuer.
- **Utility** — a bounty marketplace for real user-submitted CNF, sharing the same solver
  infrastructure. If it stalls or is captured, the chain is unaffected.

Difficulty is decoupled from SAT hardness (heavy-tailed near the phase transition) by targeting
`H(instance‖assignment)` instead, letting the law of large numbers stabilize block time.

**Its central risk differs in kind.** With hash-based PoW everyone runs the same optimal algorithm
and competes on hardware. With SAT, a privately-held 10×-faster solver is a 10× mining advantage —
and a secret algorithm is *less* accessible than an ASIC, which you can at least buy. Whether that
is a fatal centralization vector or the most valuable property of the design (a standing economic
incentive for solver research) is an open judgment call, stated rather than buried.

---

## The shared lever: cache contention

`bench/contention.py` — measures why **N single-core machines beat one N-core machine**.

With a per-instance working set of ≥2 MB, threads evict each other from shared L3:

| Working set / thread | 4 threads | 12 threads |
|---|---|---|
| 256 KB | 95% ✅ | 41% |
| **2 MB** | **68%** ⚠️ | 30% |
| **16 MB** | **62%** ⚠️ | 28% |

*(scaling efficiency = total throughput ÷ N × single-thread)*

The clean signal is the 1→4 range, where all threads land on this machine's 4 P-cores:
**≥2 MB working set costs 32–38% at only 4 threads.** The 12-thread column is inflated by
E-core scheduling and laptop thermal throttling and should be re-measured on homogeneous silicon.

This independently lands on the same **2–4 MB sweet spot** RandomX uses — large enough to fill one
core's L2, small enough that a 32-core server with 32 MB of L3 thrashes at 8+ threads.

## Measurement methodology (learned the hard way)

1. **Pin to a P-core.** Hybrid CPUs migrate threads; E-cores are 2–3× slower. Unpinned, the same
   function measured **183 ns and 527 ns** on consecutive runs. `SetThreadAffinityMask` via ctypes
   **silently returns 0** unless you declare `argtypes`/`restype`.
2. **Take the minimum, not the mean.** Standard for latency; filters scheduler and interrupt noise.
3. **Measure the clock, don't trust the spec.** Derive it from a known-latency dependency chain.
   Contradictory readings (2.82 and 5.50 GHz on a 4.5 GHz part) are the signal that a thread moved.
4. **`fastmath=False` is mandatory.** PoW requires bit-exact determinism; letting the compiler
   reassociate floating point changes rounding.

## Also here: a Wesolowski VDF

`vdf/` — a complementary primitive, not a competing design. Where the three above are probabilistic
PoW (verification means recomputing), a VDF *proves elapsed sequential time* and verifies in
milliseconds — the right tool for time-locks, sealed-bid auctions, and randomness beacons.

Measured (T = 10⁶ sequential squarings, 1024-bit modulus):

| | naive | checkpointed + parallel |
|---|---|---|
| eval (strictly serial) | 2101 ms | 2101 ms |
| prove | 11827 ms | **3823 ms** (3.1×) |
| verify | **4.83 ms** | 4.83 ms |

`eval` cannot be parallelized by anyone — that is the point. `prove` *can* be, and v2 exploits it.

## Status

| Design | Killing experiment | Done? |
|---|---|---|
| NarrowNet | **Real CUDA measurement** — the 420 ns/step GPU figure is still an estimate | ❌ needs a discrete GPU |
| PoRT | Network simulator: does latency actually dominate? | ❌ |
| SAT-PoUW | **Multi-thread scaling of MiniSat / CaDiCaL** — if near-linear, the design dies | ❌ cheapest and most decisive |

Every CPU-vs-GPU number in this repo rests on an *estimated* GPU cost. This project has already
demonstrated that estimates can be wrong by 15×. Treat them as provisional.

## Usage

```bash
pip install numpy numba
python src/narrownet_v3.py    # implementation + self-calibration
python bench/bench_v2.py      # step-cost benchmark (P-core pinned)
python bench/contention.py    # multi-thread cache contention
```

## Prior work

| | Relation |
|---|---|
| [RandomX](https://github.com/tevador/RandomX) | Same goal, different method. Its 2 GB dataset and 2 MB scratchpad thresholds are both independently reproduced here. |
| [Argon2](https://github.com/P-H-C/phc-winner-argon2) | Sequential memory-hard filling; PoRT's pool-fill borrows the idea. |
| [Primecoin](https://github.com/primecoin/primecoin) | One of the few PoUW schemes that actually worked; narrow utility. |
| [Coin.AI](https://arxiv.org/pdf/1903.09800) | PoUW via DNN *training* — hard to verify cheaply, and GPU-favoring. |
| [Anubis](https://github.com/TecharoHQ/anubis) | The anti-scraper application these could serve; currently SHA-256, hence ASIC-bypassable. |
| [drand](https://github.com/drand/drand) | Architectural reference for PoRT: featherweight nodes, tiny bandwidth, latency and liveness decide everything. |
| **Qubic** | ❌ **Not a usable reference despite appearances.** Its computors need **1 TB RAM on bare-metal UEFI** (no VPS possible), it is Anti-Military-Licensed, and its mining moved to GPUs. |

## License

MIT

---

<a name="中文说明"></a>

# 中文说明

三个设计，围绕同一个问题：**共识计算能不能做到让 CPU 和小机器胜过 GPU——如果能，代价是什么？**

这是一个实验室，不是产品。每个设计都推进到能识别出「哪个实验能一票否决它」的程度，并把那个实验明确写出来。其中一个已经开火了：这里有个成本模型**错了 15 倍**，它推翻的结论被完整记录，而不是悄悄改掉。

## 三个设计对比

| | **NarrowNet** | **PoRT** | **SAT-PoUW** |
|---|---|---|---|
| **稀缺资源** | 内存容量 | 网络位置 | 求解器质量 |
| **反 GPU 手段** | 人为构造（FP64/串行链/独占池） | 网络延迟主导关键路径 | **问题天生如此** |
| **验证成本** | 完整重算 | ~640 ms | ✅ **O(n)，代入检查** |
| **工作有用吗** | ❌ | ❌ | ✅ **有真实产出** |
| **实测了吗** | ✅ 已标定 | ❌ | ❌ |
| **什么能否定它** | 池子 <1GB 则 GPU 赢 | 按 IP 限流信号太薄 | 求解器多核线性扩展 |
| **核心风险** | 每实例需 ≥1GB | 批发 IP（/24）攻击 | **秘密求解器 = 挖矿优势** |

**三者的演进方向才是最有意思的部分**：NarrowNet 和 PoRT 费很大力气去**制造**"GPU 不适感"；而 SAT 求解**天生就是那个形状**——二十年 GPU SAT 研究的失败，本身就是安全性证明。

## 共用杠杆：缓存争用

`bench/contention.py` 量化了**为什么 N 台单核机器胜过一台 N 核机器**：

| 每线程工作集 | 4 线程效率 | 12 线程效率 |
|---|---|---|
| 256 KB | 95% ✅ | 41% |
| **2 MB** | **68%** ⚠️ | 30% |
| **16 MB** | **62%** ⚠️ | 28% |

干净信号在 1→4 区间（都落在这台机器的 4 个 P 核上）：**工作集 ≥2MB 时，仅 4 线程就损失 32–38%。** 12 线程那列被大小核调度和笔记本温度墙放大了，需在同构服务器上重测。

这独立落在了 RandomX 采用的 **2–4 MB 甜点**上——大到能塞满单核 L2，小到让 32 核服务器（32MB L3）在 8 线程以上就开始踩踏。

## 标定方法学（踩坑换来的）

1. **必须绑 P 核** —— 混合架构会迁移线程，E 核慢 2–3 倍。未绑核时同一函数连续测出 **183 ns 和 527 ns**。`SetThreadAffinityMask` 经 ctypes 调用**必须声明 argtypes/restype**，否则静默返回 0
2. **取最小值不取平均** —— 测延迟的标准做法
3. **频率要实测，别信标称** —— 用已知延迟的依赖链反推；矛盾读数（4.5GHz 的芯片测出 2.82 和 5.50）就是线程被迁移的信号
4. **`fastmath=False` 是强制的** —— PoW 需要逐位确定性，编译器重排浮点会改变舍入

## 当前状态

| 设计 | 一票否决实验 | 做了吗 |
|---|---|---|
| NarrowNet | **真实 CUDA 实测** —— 420 ns/步仍是估算 | ❌ 需要有独显的机器 |
| PoRT | 网络模拟器：延迟是否真能主导 | ❌ |
| SAT-PoUW | **MiniSat / CaDiCaL 的多线程扩展效率** —— 若接近线性则设计死亡 | ❌ 最便宜也最致命 |

**本仓库所有 CPU-vs-GPU 数字都建立在一个「估算的」GPU 成本上。本项目已经证明估算能错 15 倍。请视为暂定。**

## 许可

MIT
