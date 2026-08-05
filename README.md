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
| **Measured?** | ✅ Calibrated | ❌ No | ✅ **Measured — see below** |
| **What would kill it** | Pool < 256 MB → GPU wins | Per-IP quota signal too thin | Solvers scale linearly across cores |
| **Central risk** | Needs ≥256 MB per instance | IP wholesale (/24) attack | **Secret solver = mining advantage** |

> **Update (2026-08-05): SAT-PoUW's killing experiment has fired.** Measured on a 16-core
> Ryzen 9 7950X, CDCL solving yields a small-machine advantage of only **1.32×**, against
> **3.03×** for NarrowNet's manufactured contention on the same machine. Details and the two
> self-corrections it took to get there: **[results-7950x.zh.md](docs/results-7950x.zh.md)**.

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

#### Measured, finally — RTX 4090 vs Ryzen 9 7950X, same machine

`bench/narrownet_cuda.py`. The table above was an *estimate*; this one is not. The kernel is
deliberately written to favour the GPU (8 activations as scalars so they stay in registers, all 64
multiply-adds unrolled, both `tpb=1` and `tpb=32` tried and the faster kept), and the CPU side
reuses `src/narrownet_v3.py`'s `main_chain` rather than a reimplementation.

| Pool | GPU chains | CPU/GPU throughput | per dollar | per watt | |
|---|---|---|---|---|---|
| 2 MB | 8192 | 0.29× | 0.84× | 0.26× | ❌ GPU wins |
| 16 MB | 1036 | 0.37× | 1.06× | 0.21× | ❌ GPU wins |
| **64 MB** | 259 | **1.02×** | 2.98× | 0.57× | ~ tie |
| **256 MB** | 64 | **2.81×** | 8.19× | 1.31× | ✅ |
| **1 GB** | 16 | **6.38×** | 18.57× | 2.76× | ✅ |

**The estimate was wrong a second time — now in the opposite direction.** The corrected model
predicted the GPU winning 4.5× at 64 MB; measured, it is a dead heat. At 256 MB the CPU is ahead by
2.81×. **The ≥1 GB requirement above is retracted; 256 MB is enough**, which makes the design far
more practical (browser instances, memory per miner).

⚠️ **The GPU was given 16.2 GB, not 24 GB** — `VRAM_BUDGET = 0.72` of the 22.5 GB free on a live
desktop, which is what sets every chain count in the table above. A headless card gets ~1.48× more
chains, and these are latency-bound independent chains that scale nearly linearly in chain count at
these occupancies. Deflating by that factor: **64 MB flips to the GPU (~0.7×)** and 256 MB stays a
CPU win (~1.9×). So the defensible claim is **256 MB, not a 64 MB crossover** — the dead heat at
64 MB holds only against a GPU that is also driving a display.

Two further results:

- **Strict IEEE is load-bearing, and it is nearly free for the GPU.** Letting the compiler fuse
  `a + b*c` into an FMA changes the result — max relative error **18.9**, i.e. a completely
  different chain, so an FMA-using GPU miner produces *invalid* solutions. Only the
  `dadd_rn`/`dmul_rn` variant reproduces the CPU bit-for-bit. But forcing it costs the GPU just
  **1.09×**. H3 is a correctness constraint, not an economic moat — the design's hope that strict
  IEEE would itself be expensive does not survive measurement.
- **Per-watt is the weak flank.** The CPU only wins on energy above 256 MB (1.31×); at 16 MB the
  GPU is 4.8× more efficient. Anyone paying for electricity rather than hardware sees a different
  crossover than the per-dollar column suggests.

*(1 GB row caveat: 31 GB of system RAM only allowed 6 CPU threads at that pool size, against 16 GPU
chains in the 16.2 GB budget. The 6.38× is measured at 6 threads; extrapolating per-chain latency to
all 16 cores gives 17× as an upper bound that ignores the extra contention more threads would add.
Both sides are capacity-capped here — 31 GB of RAM holds 31 chains, 24 GB of VRAM holds 24 — which
is the actual mechanism at large pools: the 4090's 16384 cores are irrelevant when only ~20 pools
fit, so the contest reduces to per-chain latency, where the CPU is 22× faster.)*

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

**Measured — and the killing experiment fired.** Independent Cadical processes, all solving an
*identical* fixed instance (the phase transition is heavy-tailed, so varying the instance across
concurrency levels measures nothing but luck — the first attempt did exactly that and produced
garbage), on a 16-core 7950X:

| @ 16 physical cores | scaling efficiency | small-machine advantage |
|---|---|---|
| pure compute (true baseline) | 96% | 1.04× |
| **SAT (Cadical, 5.2 MB working set)** | **76%** | **1.32×** |
| NarrowNet chain (4 MB) | 33% | **3.03×** |

CDCL *does* contend — it loses 20 points against a register-only baseline. But 1.32× does not pay
for itself: one 16-core machine costs far less than sixteen single-core ones. The design fails
**quantitatively**, not qualitatively.

The obvious rebuttal — "the instance was small enough to fit in cache" — was tested and does not
hold: Cadical's peak working set at that instance is **5.18 MB**, *larger* than the 4 MB at which
the NarrowNet chain collapses to 33% on the same machine. The difference is access pattern
(whole-pool pointer chasing vs. CDCL's temporal locality), not working-set size.

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

### Re-measured on homogeneous silicon (7950X, 16C/32T, 32 MB L3 per CCD × 2)

`bench/contention_7950x.py` — same `chain()` kernel byte-for-byte, so the numbers are comparable:

| working set / thread | 4 | 8 | **16** | 32 |
|---|---|---|---|---|
| 256 KB | 97% | 90% | 75% | 44% |
| 2 MB | 96% | 80% | 63% | 37% |
| **4 MB** | 87% | **64%** | **33%** | 23% |
| 16 MB | 65% | 53% | 33% | 21% |

**The sweet spot moved: 4 MB here, not the 2 MB the laptop found.** 8 threads × 4 MB = 32 MB is
exactly one CCD's L3, and that is where the cliff appears. The laptop's 2 MB was set by *its* 12 MB
L3 across 4 P-cores; on Zen 4 the threshold scales with L3 in the same proportion.

That is worth stating plainly: **this lever is a function of the target's cache topology, not a
portable constant.** A design that hard-codes a working-set size is tuning to one generation of one
vendor's cache hierarchy — which is a real problem for anything that wants to be a security
assumption.

This still lands near the **2–4 MB** region RandomX uses — large enough to fill one core's L2, small
enough that a many-core server thrashes once the per-CCD L3 is oversubscribed.

## Measurement methodology (learned the hard way)

1. **Pin to a P-core.** Hybrid CPUs migrate threads; E-cores are 2–3× slower. Unpinned, the same
   function measured **183 ns and 527 ns** on consecutive runs. `SetThreadAffinityMask` via ctypes
   **silently returns 0** unless you declare `argtypes`/`restype`.
2. **Take the minimum, not the mean.** Standard for latency; filters scheduler and interrupt noise.
3. **Measure the clock, don't trust the spec.** Derive it from a known-latency dependency chain.
   Contradictory readings (2.82 and 5.50 GHz on a 4.5 GHz part) are the signal that a thread moved.
4. **`fastmath=False` is mandatory.** PoW requires bit-exact determinism; letting the compiler
   reassociate floating point changes rounding.
5. **Hold the work constant, not just the parameters.** Random 3SAT at the phase transition is
   heavy-tailed: the same size and ratio varies by tens of times across seeds. Handing different
   concurrency levels different instances measures instance luck, not scaling — calibration said
   13.7 s and the run measured 0.32 s. Pin one instance, verify its spread is under 1.25×, and give
   every worker that same one.
6. **Verify that your control is actually a control.** A CPython SHA-256 loop was used as the
   "zero working set" baseline; it allocates a fresh bytes object every iteration, and it reported
   75% where a register-only chain reports 96%. Every conclusion drawn by comparing against it was
   wrong by that gap.
7. **Test the explanation that explains everything.** All-core downclocking was the natural account
   of the 16-thread loss — until frequency was measured directly from a known-latency dependency
   chain and came back at **98%**. An explanation that fits every number is the one most worth
   measuring separately.

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
| NarrowNet | **Real CUDA measurement** — replacing the estimated 420 ns/step | ✅ **survived — 256 MB suffices** |
| PoRT | Network simulator: does latency actually dominate? | ❌ |
| SAT-PoUW | **Multi-thread scaling of MiniSat / CaDiCaL** — if near-linear, the design dies | ✅ **fired — 1.32× is not enough** |

Two of the three have now been run, on a 7950X + RTX 4090. **NarrowNet survived its own killing
experiment and got cheaper** (256 MB suffices, not 1 GB); **SAT-PoUW did not survive its own.**
PoRT remains untested, and its known quota gap (§8.1) is still open.

The repo's standing warning has now fired twice in both directions: the original cost model was
wrong by 15× (optimistic), and its correction was wrong by ~4× (pessimistic). **Estimates here have
never once survived measurement.** Numbers not yet marked as measured should be read accordingly.

## Usage

```bash
pip install numpy numba python-sat psutil

python src/narrownet_v3.py         # implementation + self-calibration
python bench/bench_v2.py           # step-cost benchmark (P-core pinned)
python bench/contention.py         # cache contention (original, hybrid laptop CPU)

# the 7950X round — see docs/results-7950x.zh.md
python bench/sat_scaling_local.py  # SAT-PoUW killing experiment (fixed instance)
python bench/fp_quick.py           # Cadical's real working set
python bench/contention_7950x.py   # contention, re-measured on homogeneous silicon
python bench/verify_baseline.py    # measured clocks — refutes the downclocking story
python bench/true_baseline.py      # register-only control, same process harness
```

Raw data: `results_7950x.json`, `contention_7950x.json`, `sat_footprint.json`,
`freq_7950x.json`, `true_baseline_7950x.json`.

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
| **实测了吗** | ✅ 已标定 | ❌ | ✅ **已实测，见下** |
| **什么能否定它** | 池子 <256MB 则 GPU 赢 | 按 IP 限流信号太薄 | 求解器多核线性扩展 |
| **核心风险** | 每实例需 ≥256MB | 批发 IP（/24）攻击 | **秘密求解器 = 挖矿优势** |

> **更新（2026-08-05）：SAT-PoUW 的一票否决实验已经开火。** 在 16 核 7950X 上实测，
> CDCL 求解只能带来 **1.32×** 的小机器优势，而同一台机器上 NarrowNet 人为构造的争用
> 能到 **3.03×**。全过程（含两次自我纠错）：**[results-7950x.zh.md](docs/results-7950x.zh.md)**

### SAT-PoUW 实测结果

| @ 16 物理核 | 扩展效率 | 小机器优势 |
|---|---|---|
| 纯计算（真基线） | 96% | 1.04× |
| **SAT（Cadical，工作集 5.2MB）** | **76%** | **1.32×** |
| NarrowNet 链（4MB） | 33% | **3.03×** |

CDCL **确实有**争用（比纯寄存器基线低 20 个百分点），但 1.32× 换不来经济优势：
一台 16 核机器远比 16 台单核机器便宜。**这是定量否决，不是"SAT 毫无争用"。**

对该结论最强的反驳——"实例太小塞进缓存了"——已被证伪：Cadical 在该实例上的峰值工作集是
**5.18MB**，**比 NarrowNet 崩到 33% 的 4MB 还大**。差异在访问模式（全池指针追逐
vs CDCL 的时间局部性），不在工作集大小。

**三者的演进方向才是最有意思的部分**：NarrowNet 和 PoRT 费很大力气去**制造**"GPU 不适感"；而 SAT 求解**天生就是那个形状**——二十年 GPU SAT 研究的失败，本身就是安全性证明。

## 共用杠杆：缓存争用

`bench/contention.py` 量化了**为什么 N 台单核机器胜过一台 N 核机器**：

| 每线程工作集 | 4 线程效率 | 12 线程效率 |
|---|---|---|
| 256 KB | 95% ✅ | 41% |
| **2 MB** | **68%** ⚠️ | 30% |
| **16 MB** | **62%** ⚠️ | 28% |

干净信号在 1→4 区间（都落在这台机器的 4 个 P 核上）：**工作集 ≥2MB 时，仅 4 线程就损失 32–38%。** 12 线程那列被大小核调度和笔记本温度墙放大了，需在同构服务器上重测。

### 同构硅复测（7950X，16C/32T，32MB L3 × 2 CCD）

`bench/contention_7950x.py`——`chain()` 计算核心逐字未改，数字可直接对比：

| 工作集/线程 | 4 | 8 | **16** | 32 |
|---|---|---|---|---|
| 256 KB | 97% | 90% | 75% | 44% |
| 2 MB | 96% | 80% | 63% | 37% |
| **4 MB** | 87% | **64%** | **33%** | 23% |
| 16 MB | 65% | 53% | 33% | 21% |

**甜点变了：这里是 4MB，不是笔记本上的 2MB。** 8 线程 × 4MB = 32MB 正好是单个 CCD 的 L3，
断崖就出现在那里。笔记本的 2MB 是被它 4 个 P 核共享 12MB L3 决定的；换到 Zen4，阈值随 L3 等比例上移。

这一点值得直说：**这条杠杆是目标芯片缓存拓扑的函数，不是可移植的常数。**
把工作集大小写死的设计，等于在给某一代某一家的缓存层级做调参——对想拿它当安全假设的东西来说，这是真问题。

数值仍落在 RandomX 采用的 **2–4MB** 区间附近——大到能塞满单核 L2，小到让多核服务器在单 CCD 的 L3 被超额订阅后开始踩踏。

## 标定方法学（踩坑换来的）

1. **必须绑 P 核** —— 混合架构会迁移线程，E 核慢 2–3 倍。未绑核时同一函数连续测出 **183 ns 和 527 ns**。`SetThreadAffinityMask` 经 ctypes 调用**必须声明 argtypes/restype**，否则静默返回 0
2. **取最小值不取平均** —— 测延迟的标准做法
3. **频率要实测，别信标称** —— 用已知延迟的依赖链反推；矛盾读数（4.5GHz 的芯片测出 2.82 和 5.50）就是线程被迁移的信号
4. **`fastmath=False` 是强制的** —— PoW 需要逐位确定性，编译器重排浮点会改变舍入

## 当前状态

| 设计 | 一票否决实验 | 做了吗 |
|---|---|---|
| NarrowNet | **真实 CUDA 实测** —— 取代估算的 420 ns/步 | ✅ **活下来了：256MB 就够** |
| PoRT | 网络模拟器：延迟是否真能主导 | ❌ |
| SAT-PoUW | **MiniSat / CaDiCaL 的多线程扩展效率** —— 若接近线性则设计死亡 | ✅ **已开火：1.32× 不够** |

三个里已经做掉两个（7950X + RTX 4090）。**NarrowNet 扛过了自己的一票否决，而且变便宜了**
（256MB 就够，不必 1GB）；**SAT-PoUW 没扛过。** PoRT 仍未测，它已知的配额漏洞（§8.1）也还开着。

本仓库那句警告已经在两个方向上各应验一次：最初的成本模型错了 15 倍（乐观），
而它的修正版又错了约 4 倍（悲观）。**这里的估算至今没有一次经受住实测。**
凡是还没标注「已实测」的数字，请照此对待。

### NarrowNet CUDA 实测（RTX 4090 vs 7950X，同机）

`bench/narrownet_cuda.py`。kernel 刻意写得偏袒 GPU（8 个激活值用标量保证进寄存器、
64 次乘加全展开、tpb=1 和 32 都跑取快的），CPU 侧直接复用 `src/narrownet_v3.py` 的
`main_chain` 不另写。

| 池子 | GPU链数 | CPU/GPU 吞吐 | 每美元 | 每瓦 | |
|---|---|---|---|---|---|
| 2 MB | 8192 | 0.29× | 0.84× | 0.26× | ❌ GPU 赢 |
| 16 MB | 1036 | 0.37× | 1.06× | 0.21× | ❌ GPU 赢 |
| **64 MB** | 259 | **1.02×** | 2.98× | 0.57× | ~ 打平 |
| **256 MB** | 64 | **2.81×** | 8.19× | 1.31× | ✅ |
| **1 GB** | 16 | **6.38×** | 18.57× | 2.76× | ✅ |

**估算第二次出错，这回是反方向。** 修正后的模型预测 64MB 时 GPU 赢 4.5×，实测是打平；
256MB 时 CPU 已经赢 2.81×。**上面「必须 ≥1GB」的结论予以撤回，256MB 就够** —— 这让设计
实用得多（浏览器实例、每矿工内存占用）。

⚠️ **给 GPU 的是 16.2GB，不是 24GB。** 脚本里 `VRAM_BUDGET = 0.72`，只取了桌面在用时空闲的
22.5GB 的 72% —— 上表每一个链数都是这么来的。无显示输出的卡能多拿约 **1.48 倍**链数，而这些
是延迟受限的独立链，在当前占用率下吞吐随链数近似线性。按这个系数折算：**64MB 翻回给 GPU
（约 0.7×）**，256MB 仍是 CPU 赢（约 1.9×）。所以站得住的说法是 **256MB，而不是「拐点在
64MB」** —— 64MB 的打平只在对手那张卡同时还在驱动显示器时成立。

另外两个结果：

- **严格 IEEE 是承重的，但对 GPU 几乎免费。** 让编译器把 `a + b*c` 合并成 FMA 会改变结果，
  最大相对误差 **18.9**，即完全不同的链 —— 用 FMA 的 GPU 矿工产出的是**无效解**。
  只有 `dadd_rn`/`dmul_rn` 变体能与 CPU 逐位相同。但强制它只让 GPU 慢 **1.09×**。
  H3 是正确性约束，不是经济护城河 —— 设计原本指望"严格 IEEE 本身很贵"，实测不成立。
- **每瓦是软肋。** CPU 只在 256MB 以上才赢能效（1.31×）；16MB 时 GPU 能效是 CPU 的 4.8 倍。
  付电费而不是付硬件钱的人，看到的拐点和「每美元」那列不是一回事。

*(1GB 那行注意：31GB 系统内存只够开 6 个 CPU 线程，对上 16.2GB 预算里的 16 条 GPU 链。
6.38× 是 6 线程实测；按每链延迟外推到满 16 核得 17×，那是忽略了更多线程额外争用的上界。
这里**两边都被容量卡死** —— 31GB 内存装 31 条链，24GB 显存装 24 条 —— 这才是大池子时的真实
机制：只塞得下约 20 个池子的时候，4090 的 16384 个核心毫无意义，比赛退化成单链延迟之争，
而 CPU 在这上面快 22 倍。)*

## 方法学补充（本轮踩坑换来的）

5. **要固定的是"工作量"，不只是参数。** 相变点随机 3SAT 是重尾分布：同规模同比例，换个种子差几十倍。
   给不同并发度喂不同实例，测到的是实例运气不是扩展性——标定说 13.7s，实测 0.32s。
   正确做法：固定一个实例，先验证它重复测量的离散度 < 1.25×，再让所有 worker 都解它。
6. **必须验证你的"对照组"真的是对照。** 曾用 CPython 的 SHA-256 循环当"零工作集"基线，
   但它每次迭代都分配新 bytes 对象；它测出 75%，而纯寄存器依赖链测出 96%。
   所有靠它做的比较都错了这个差值。
7. **越是"什么都能解释"的解释，越要单独测。** 全核降频本来能自然解释 16 线程的损失——
   直到用已知延迟的依赖链直接实测频率，得到 **98%**。

## 许可

MIT
