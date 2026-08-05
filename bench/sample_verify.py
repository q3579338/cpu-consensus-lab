#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽查式验证原型 —— 回答一个判决性问题:

    NarrowNet 的验证能否从"重算一次尝试"(256MB 池实测 577ms = 填池 382 + 跑链 195)
    降到毫秒级,而矿工附带的证明只有几百 KB?

若证明大小压不住 => 这条路(乐观验证 + 抽查)当场否掉。

被测变体:**v4-immutable(池子在跑链阶段只读)**
    v3 的链每步写回 pool[base]。写回意味着验证第 i 段需要"第 i 段开始时的池子状态",
    矿工必须维护每个检查点时刻的池默克尔根 —— 增量维护 ≈ 每步 ~20 次哈希 × 10^6 步
    ≈ 2×10^7 次哈希,比跑链本身贵一个量级;更糟的是这份哈希簿记可以离线批量重放,
    是 GPU 友好的,会稀释整个设计的反 GPU 性。去掉写回后全程只需**一个**池根。
    (写回在 v3 里的作用只是废掉 GPU 的只读数据缓存 —— 一个次要减速项,
     值不值得用"验证成本×10"去换,由本实验和后续 CUDA 消融来回答。)

结构性质(决定证明大小的两个关键事实):
  * 填池每格只写一次、链阶段不再改 => 填池期间任何"已写区域"的回读值
    就是**最终池子里的值**,直接对同一个池根开默克尔证明;
    读到未写区域(j >= i)验证者自己知道是 0.0,**零字节证明**。
  * 抓到"跳过 f 比例工作"的概率 = 1-(1-f)^k,只与抽样次数 k 有关、
    与段长无关 => 段尽量切短,证明尽量小。

协议(非交互,Fiat-Shamir):
  矿工提交: nonce, 池根 R_pool, 填池PRNG检查点根 R_fill, 链激活检查点根 R_act, 最终digest
  挑战     = sha256(以上全部)  -> 采样 k_fill 个填池段 + k_chain 个链段
  证明     = 被抽中段的检查点开启 + 段内每步读到的 64-double 块 + 各自默克尔路径
             (整个证明打包成"节点集合",跨条目去重 —— 两种口径都统计)
  验证     = 用与矿工完全相同的 jitted 内核重放被抽中的段(逐位一致由共享代码保证),
             检查:重放导出的块地址 == 证明声称的地址;Merkle 路径成立;
             段终点 == 下一个检查点;最终digest所在段封顶。
"""
import hashlib
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.narrownet_v3 import BLK, M64, TO_INT, W  # noqa: E402

# ---------------- 参数 ----------------
POOL_LOG2 = 25            # 2^25 double = 256 MB
DEPTH = 1_000_000
C_CK = 8                  # 链激活检查点间隔(=链抽样段长, 步)
F_SPACE = 4096            # 填池PRNG检查点间距(可稀疏:填池作弊近乎全有全无)
F_LEN = 32                # 填池抽样窗长(从检查点起重放这么多元素)
K_CHAIN = 20              # 链抽样段数   (抓 30% 跳步概率 99.92%)
K_FILL = 8                # 填池抽样段数

try:                       # Merkle 哈希:有 blake3 用 blake3(约快一个量级)
    from blake3 import blake3 as _b3
    H32 = lambda b: _b3(b).digest()
    HASH_NAME = "blake3"
except ImportError:
    H32 = lambda b: hashlib.sha256(b).digest()
    HASH_NAME = "sha256"


# ---------------- numba 内核(矿工与验证者共享 => 逐位一致免证) ----------------

@njit(cache=True, fastmath=False)
def fill_ckpt(pool, s0, s1, mask, f_ck, cx, cy):
    """与 v3 fill_pool 逐字相同,只是每 f_ck 个元素记录一次 (x,y)。"""
    n = pool.size
    x = s0
    y = s1
    for g in range(n // f_ck):
        cx[g] = x
        cy[g] = y
        base_i = g * f_ck
        for kk in range(f_ck):
            i = base_i + kk
            t = x
            t ^= (t << np.uint64(23)) & M64
            t ^= t >> np.uint64(17)
            t ^= y ^ (y >> np.uint64(26))
            x = y
            y = t
            v = (x + y) & M64
            j = int(v & mask)
            v = (v ^ np.uint64(int(abs(pool[j]) * TO_INT))) & M64
            v = (v * np.uint64(0x9E3779B97F4A7C15)) & M64
            pool[i] = (v >> np.uint64(11)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0
    cx[n // f_ck] = x            # 末尾哨兵检查点
    cy[n // f_ck] = y


@njit(cache=True, fastmath=False)
def fill_segment(seg_start, length, mask, x, y, ext_vals, out_vals, out_ext_j):
    """
    重放填池的一段 [seg_start, seg_start+length)。

    读 pool[j] 的三种来源:
      j >= i          -> 0.0(未写区域,免证明)
      seg_start<=j<i  -> 本段刚算出的值(免证明)
      j < seg_start   -> 外部值,按出现顺序从 ext_vals 消费(矿工附带+默克尔证明)
    返回消费的外部读地址序列(供验证者对照默克尔证明的声称地址)。
    矿工提取证明与验证者重放走的是**同一个函数**:矿工先用全池预跑一遍拿到
    外部读序列,再以 ext_vals 喂回本函数确认一致。
    """
    n_ext = 0
    for k in range(length):
        i = seg_start + k
        t = x
        t ^= (t << np.uint64(23)) & M64
        t ^= t >> np.uint64(17)
        t ^= y ^ (y >> np.uint64(26))
        x = y
        y = t
        v = (x + y) & M64
        j = int(v & mask)
        if j >= i:
            pj = 0.0
        elif j >= seg_start:
            pj = out_vals[j - seg_start]
        else:
            pj = ext_vals[n_ext]
            out_ext_j[n_ext] = j
            n_ext += 1
        v = (v ^ np.uint64(int(abs(pj) * TO_INT))) & M64
        v = (v * np.uint64(0x9E3779B97F4A7C15)) & M64
        out_vals[k] = (v >> np.uint64(11)) * (1.0 / 9007199254740992.0) * 2.0 - 1.0
    return n_ext, x, y


@njit(cache=True, fastmath=False)
def fill_extract_reads(pool, seg_start, length, mask, x, y, ext_vals, out_ext_j):
    """矿工侧:预跑一段,从真池子里收集外部读的 (地址, 值)。"""
    n_ext = 0
    for k in range(length):
        i = seg_start + k
        t = x
        t ^= (t << np.uint64(23)) & M64
        t ^= t >> np.uint64(17)
        t ^= y ^ (y >> np.uint64(26))
        x = y
        y = t
        v = (x + y) & M64
        j = int(v & mask)
        if j < seg_start:
            ext_vals[n_ext] = pool[j]
            out_ext_j[n_ext] = j
            n_ext += 1
        pj = pool[j] if j < i else 0.0
        v = (v ^ np.uint64(int(abs(pj) * TO_INT))) & M64
        # 无需写值 —— 池子已填好,这里只收集读序列
    return n_ext, x, y


@njit(cache=True, fastmath=False)
def chain_ro_ckpt(pool, h, depth, blk_mask, c_ck, acts):
    """v3 main_chain 去掉写回(池子只读),每 c_ck 步记录激活检查点。"""
    a = h.copy()
    tmp = np.empty(W, dtype=np.float64)
    for step in range(depth):
        if step % c_ck == 0:
            for r in range(W):
                acts[step // c_ck, r] = a[r]
        acc = np.uint64(0xcbf29ce484222325)
        for j in range(W):
            acc = ((acc ^ np.uint64(int(abs(a[j]) * TO_INT))) * np.uint64(0x100000001b3)) & M64
        base = int(acc & blk_mask) * BLK
        for r in range(W):
            s = a[r]
            off = base + r * W
            for c in range(W):
                s += pool[off + c] * a[c]
            tmp[r] = s
        s2 = 0.0
        for r in range(W):
            v = tmp[r]
            v = v / (1.0 + abs(v)) if v > 0.0 else 0.05 * v
            a[r] = v
            s2 += v * v
        inv = 1.0 / (np.sqrt(s2) + 1e-12)
        for r in range(W):
            a[r] *= inv
    for r in range(W):
        acts[depth // c_ck, r] = a[r]
    return a


@njit(cache=True, fastmath=False)
def chain_replay(a0, steps, blk_mask, blocks, out_bases):
    """
    重放链的一段:激活从 a0 出发,第 k 步的 64-double 权重块由 blocks[k] 提供。
    返回段终激活;out_bases 记录每步由激活导出的块索引(验证者拿它对照证明声称的地址)。
    矿工侧提取时以真池子切片喂 blocks,验证者以证明里的块喂 —— 同一函数,逐位一致。
    """
    a = a0.copy()
    tmp = np.empty(W, dtype=np.float64)
    for k in range(steps):
        acc = np.uint64(0xcbf9ce484222325 * 0 + 0xcbf29ce484222325)
        for j in range(W):
            acc = ((acc ^ np.uint64(int(abs(a[j]) * TO_INT))) * np.uint64(0x100000001b3)) & M64
        out_bases[k] = int(acc & blk_mask)
        for r in range(W):
            s = a[r]
            off = r * W
            for c in range(W):
                s += blocks[k, off + c] * a[c]
            tmp[r] = s
        s2 = 0.0
        for r in range(W):
            v = tmp[r]
            v = v / (1.0 + abs(v)) if v > 0.0 else 0.05 * v
            a[r] = v
            s2 += v * v
        inv = 1.0 / (np.sqrt(s2) + 1e-12)
        for r in range(W):
            a[r] *= inv
    return a


# ---------------- Merkle(sha256, 叶=任意 bytes) ----------------

def merkle_build(leaves_bytes):
    """返回 levels: levels[0]=叶哈希 ... levels[-1]=[root]。奇数结尾自提升。"""
    lvl = [H32(b) for b in leaves_bytes]
    levels = [lvl]
    while len(lvl) > 1:
        nxt = []
        for i in range(0, len(lvl) - 1, 2):
            nxt.append(H32(lvl[i] + lvl[i + 1]))
        if len(lvl) % 2:
            nxt.append(lvl[-1])
        lvl = nxt
        levels.append(lvl)
    return levels


def merkle_path(levels, idx):
    """返回 [(level, sibling_idx)];配对缺席(自提升)的层跳过。"""
    path = []
    for lv in range(len(levels) - 1):
        sib = idx ^ 1
        if sib < len(levels[lv]):
            path.append((lv, sib))
        idx //= 2
    return path


def merkle_verify(root, leaf_bytes, idx, path_nodes, n_leaves):
    """path_nodes: [(level, sib_idx, 32B)]。n_leaves 是共识公开参数,
    由它逐层推出兄弟是否存在(奇数层末尾自提升,无兄弟)。"""
    h = H32(leaf_bytes)
    pi = iter(path_nodes)
    cur = idx
    size = n_leaves
    while size > 1:
        sib = cur ^ 1
        if sib < size:
            nxt = next(pi, None)
            if nxt is None:
                return False
            sh = nxt[2]
            h = H32(sh + h) if cur & 1 else H32(h + sh)
        cur //= 2
        size = (size + 1) // 2
    return h == root


class ProofBundle:
    """统计证明大小:原始口径(每条目独立)与去重口径(节点集合)。"""

    def __init__(self):
        self.raw = 0
        self.payload = 0
        self.nodes = {}          # (tree_id, level, idx) -> 32B

    def add(self, tree_id, levels, leaf_bytes, idx):
        path = merkle_path(levels, idx)
        self.raw += len(leaf_bytes) + 32 * len(path)
        self.payload += len(leaf_bytes)
        for lv, sib in path:
            self.nodes[(tree_id, lv, sib)] = levels[lv][sib]
        return [(lv, sib, levels[lv][sib]) for lv, sib in path]

    @property
    def dedup(self):
        return self.payload + 32 * len(self.nodes)


# ---------------- 主流程 ----------------

def main():
    n = 1 << POOL_LOG2
    mask = np.uint64(n - 1)
    blk_mask = np.uint64((n // BLK) - 1)
    n_leaves = n // BLK

    print(f"=== 抽查式验证原型(v4-immutable,{n * 8 // 2**20}MB 池,{DEPTH:,} 步)===")
    print(f"    链段 {C_CK} 步 × {K_CHAIN}   填池窗 {F_LEN} 元素 × {K_FILL}"
          f"(检查点间距 {F_SPACE})   Merkle={HASH_NAME}\n")

    seed = hashlib.sha256(b"lootchain-narrownet-demo").digest()
    nonce = 7
    d = hashlib.sha256(seed + nonce.to_bytes(8, "big")).digest()
    s0 = np.uint64(int.from_bytes(d[0:8], "big") | 1)
    s1 = np.uint64(int.from_bytes(d[8:16], "big") | 1)

    # ---------- 矿工侧 ----------
    pool = np.zeros(n, dtype=np.float64)
    n_fck = n // F_SPACE
    cx = np.zeros(n_fck + 1, dtype=np.uint64)
    cy = np.zeros(n_fck + 1, dtype=np.uint64)

    small = np.zeros(1 << 12, dtype=np.float64)          # JIT 预热
    fill_ckpt(small, s0, s1, np.uint64(small.size - 1), 64,
              np.zeros(small.size // 64 + 1, dtype=np.uint64),
              np.zeros(small.size // 64 + 1, dtype=np.uint64))

    t0 = time.perf_counter()
    fill_ckpt(pool, s0, s1, mask, F_SPACE, cx, cy)
    t_fill = time.perf_counter() - t0

    # sha512 而非 sha256:必须喂满 W=8 个 uint64,否则内核越界读(见 narrownet_v3 同处注释)
    h0 = np.frombuffer(hashlib.sha512(d).digest(), dtype=np.uint64)[:W].copy()
    assert h0.shape[0] == W
    h0 = (h0 >> np.uint64(11)).astype(np.float64) / 9007199254740992.0 * 2.0 - 1.0

    n_cck = DEPTH // C_CK
    acts = np.zeros((n_cck + 1, W), dtype=np.float64)
    chain_ro_ckpt(pool[: 1 << 12], h0, 64, np.uint64(((1 << 12) // BLK) - 1), C_CK,
                  np.zeros((64 // C_CK + 1, W), dtype=np.float64))   # JIT 预热

    t0 = time.perf_counter()
    a_final = chain_ro_ckpt(pool, h0, DEPTH, blk_mask, C_CK, acts)
    t_chain = time.perf_counter() - t0
    digest = hashlib.sha256(a_final.tobytes()).digest()

    # 三棵默克尔树(矿工的一次性承诺开销)
    pool_bytes = pool.tobytes()
    t0 = time.perf_counter()
    pool_lv = merkle_build([pool_bytes[i * 512:(i + 1) * 512] for i in range(n_leaves)])
    t_tree_pool = time.perf_counter() - t0
    t0 = time.perf_counter()
    fck_bytes = [cx[i].tobytes() + cy[i].tobytes() for i in range(n_fck)]
    fill_lv = merkle_build(fck_bytes)
    act_bytes = [acts[i].tobytes() for i in range(n_cck + 1)]
    act_lv = merkle_build(act_bytes)
    t_tree_ck = time.perf_counter() - t0

    r_pool, r_fill, r_act = pool_lv[-1][0], fill_lv[-1][0], act_lv[-1][0]
    print("矿工侧成本:")
    print(f"    填池 {t_fill*1e3:7.0f} ms   跑链 {t_chain*1e3:6.0f} ms   "
          f"池树 {t_tree_pool*1e3:6.0f} ms   检查点树 {t_tree_ck*1e3:5.0f} ms({HASH_NAME})")
    base_attempt = t_fill + t_chain
    overhead = (t_tree_pool + t_tree_ck) / base_attempt
    print(f"    承诺开销 = 尝试本体的 {overhead:.0%}"
          f"(树代码是纯 Python;瓶颈一半在解释器循环)\n")

    # ---------- Fiat-Shamir 采样 ----------
    challenge = hashlib.sha256(b"hdr" + r_pool + r_fill + r_act + digest
                               + nonce.to_bytes(8, "big")).digest()
    def draws(tag, count, space):
        out, ctr = [], 0
        while len(out) < count:
            hh = hashlib.sha256(challenge + tag + ctr.to_bytes(4, "big")).digest()
            v = int.from_bytes(hh[:8], "big") % space
            if v not in out:
                out.append(v)
            ctr += 1
        return out

    fill_segs = draws(b"F", K_FILL, n_fck)
    chain_segs = draws(b"C", K_CHAIN, n_cck)

    # ---------- 矿工:提取证明 ----------
    bundle = ProofBundle()
    proof_fill, proof_chain = [], []

    for g in fill_segs:
        a0 = g * F_SPACE
        ext_v = np.zeros(F_LEN, dtype=np.float64)
        ext_j = np.zeros(F_LEN, dtype=np.int64)
        n_ext, _, _ = fill_extract_reads(pool, a0, F_LEN, mask, cx[g], cy[g], ext_v, ext_j)
        item = {"g": g, "n_ext": n_ext,
                "ext": [(int(ext_j[i]), int(ext_j[i]) // BLK) for i in range(n_ext)]}
        # 证明:窗口起点 PRNG 检查点 + 窗口输出所在池叶 + 每个外部读的池叶
        #(窗口终点不再对下一检查点 —— 输出值直接对池根,绑定已足够)
        item["p_ck0"] = bundle.add("fill", fill_lv, fck_bytes[g], g)
        seg_leaf = a0 // BLK                       # F_SPACE 为 64 的倍数 => 叶对齐
        item["p_out"] = bundle.add("pool", pool_lv,
                                   pool_bytes[seg_leaf * 512:(seg_leaf + 1) * 512], seg_leaf)
        item["p_ext"] = []
        for jj, leaf in item["ext"]:
            item["p_ext"].append(bundle.add("pool", pool_lv,
                                            pool_bytes[leaf * 512:(leaf + 1) * 512], leaf))
        proof_fill.append(item)

    warm_b = np.ascontiguousarray(pool[:C_CK * BLK].reshape(C_CK, BLK))
    chain_replay(h0, C_CK, blk_mask, warm_b, np.zeros(C_CK, dtype=np.int64))  # JIT 预热

    for g in chain_segs:
        a0 = acts[g].copy()
        bases = np.zeros(C_CK, dtype=np.int64)
        blocks = np.zeros((C_CK, BLK), dtype=np.float64)
        # 提取:地址取决于当前激活,只能边走边取 —— 每步先用 chain_replay
        # 单步探出地址(块内容此时无关),再取真实块走这一步。
        aa = a0.copy()
        for k in range(C_CK):
            b1 = np.zeros(1, dtype=np.int64)
            # 先用当前激活算出本步地址(chain_replay 一步、块内容后补):
            probe = np.zeros((1, BLK), dtype=np.float64)
            chain_replay(aa, 1, blk_mask, probe, b1)     # 只取地址
            base = int(b1[0])
            blocks[k] = pool[base * BLK:(base + 1) * BLK]
            aa = chain_replay(aa, 1, blk_mask, blocks[k:k + 1], b1)
            bases[k] = base
        item = {"g": g, "bases": [int(b) for b in bases]}
        item["p_a0"] = bundle.add("act", act_lv, act_bytes[g], g)
        item["p_a1"] = bundle.add("act", act_lv, act_bytes[g + 1], g + 1)
        item["p_blk"] = [bundle.add("pool", pool_lv,
                                    pool_bytes[b * 512:(b + 1) * 512], b) for b in bases]
        item["blocks"] = blocks
        proof_chain.append(item)

    print(f"证明大小:  原始口径 {bundle.raw/1024:8.1f} KB"
          f"   去重口径 {bundle.dedup/1024:8.1f} KB")
    per_chain = sum(512 + 32 * 19 for _ in range(C_CK)) / 1024
    print(f"    (链 {K_CHAIN} 段 ≈ {K_CHAIN*per_chain:.0f} KB 是大头;"
          f"填池 {K_FILL} 段;检查点开启若干)\n")

    # ---------- 验证者 ----------
    n_act_leaves = n_cck + 1
    # 预热 fill_segment 的 JIT(不计入验证耗时)
    _w = np.zeros(F_LEN, dtype=np.float64)
    fill_segment(0, F_LEN, mask, s0, s1, np.zeros(F_LEN + 1, dtype=np.float64),
                 _w, np.zeros(F_LEN, dtype=np.int64))

    def run_verify():
        t0 = time.perf_counter()
        ok = True

        for item in proof_fill:
            g = item["g"]
            ck = fck_bytes[g]                               # 实际从证明取;字节相同
            ok &= merkle_verify(r_fill, ck, g, item["p_ck0"], n_fck)
            x = np.uint64(int.from_bytes(ck[:8], "little"))
            y = np.uint64(int.from_bytes(ck[8:], "little"))
            # 外部读值取自证明附带的叶字节(默克尔已验),不摸原池子
            ext_v = np.array(
                [np.frombuffer(pool_bytes[leaf * 512:(leaf + 1) * 512],
                               dtype=np.float64)[jj % BLK] for jj, leaf in item["ext"]]
                + [0.0], dtype=np.float64)
            out_v = np.zeros(F_LEN, dtype=np.float64)
            out_j = np.zeros(F_LEN, dtype=np.int64)
            n_ext, xf, yf = fill_segment(g * F_SPACE, F_LEN, mask, x, y, ext_v, out_v, out_j)
            ok &= (n_ext == item["n_ext"])
            for i in range(n_ext):                          # 重放导出的地址 == 声称的地址
                ok &= int(out_j[i]) == item["ext"][i][0]
            for (jj, leaf), pp in zip(item["ext"], item["p_ext"]):
                ok &= merkle_verify(r_pool, pool_bytes[leaf * 512:(leaf + 1) * 512],
                                    leaf, pp, n_leaves)
            # 窗口输出对承诺叶:本窗 F_LEN 个值必须与池叶对应半区一致
            seg_leaf = (g * F_SPACE) // BLK
            leafb = pool_bytes[seg_leaf * 512:(seg_leaf + 1) * 512]
            ok &= merkle_verify(r_pool, leafb, seg_leaf, item["p_out"], n_leaves)
            off = (g * F_SPACE) % BLK
            ok &= np.frombuffer(leafb, dtype=np.float64)[off:off + F_LEN].tobytes() \
                == out_v.tobytes()

        for item in proof_chain:
            g = item["g"]
            a0b, a1b = act_bytes[g], act_bytes[g + 1]
            ok &= merkle_verify(r_act, a0b, g, item["p_a0"], n_act_leaves)
            ok &= merkle_verify(r_act, a1b, g + 1, item["p_a1"], n_act_leaves)
            bases = np.zeros(C_CK, dtype=np.int64)
            a_end = chain_replay(np.frombuffer(a0b, dtype=np.float64).copy(),
                                 C_CK, blk_mask, item["blocks"], bases)
            for k, b in enumerate(item["bases"]):
                ok &= int(bases[k]) == b                   # 地址一致
                leafb = item["blocks"][k].tobytes()
                ok &= merkle_verify(r_pool, leafb, b, item["p_blk"][k], n_leaves)
            ok &= a_end.tobytes() == a1b                   # 段终点 == 下一检查点

        return ok, time.perf_counter() - t0

    ok, t_first = run_verify()
    ok2, t_verify = run_verify()                            # 稳态(缓存热)
    ok &= ok2
    print(f"验证结果:  {'✅ 全部通过' if ok else '❌ 失败'}   "
          f"耗时 {t_verify*1e3:.1f} ms(首次含冷缓存 {t_first*1e3:.1f} ms)")
    print(f"    对比全量重算 {base_attempt*1e3:.0f} ms  =>  加速 {base_attempt/t_verify:.0f}×\n")

    # ---------- 负面测试:篡改必须被抓 ----------
    item = proof_chain[0]
    tampered = item["blocks"].copy()
    tampered[0, 0] += 1e-9
    bases = np.zeros(C_CK, dtype=np.int64)
    a_end = chain_replay(np.frombuffer(act_bytes[item["g"]], dtype=np.float64).copy(),
                         C_CK, blk_mask, tampered, bases)
    leaf_ok = merkle_verify(r_pool, tampered[0].tobytes(), item["bases"][0],
                            item["p_blk"][0], n_leaves)
    end_ok = a_end.tobytes() == act_bytes[item["g"] + 1]
    caught = (not leaf_ok) or (not end_ok)
    print(f"负面测试:  篡改一个权重块 -> 默克尔 {'过' if leaf_ok else '断'} / "
          f"段终点 {'合' if end_ok else '偏'}  =>  {'✅ 被抓' if caught else '❌ 漏网'}")

    ck_pub = 3 * 32
    print(f"""
=== 汇总 ===
  区块头新增承诺: {ck_pub} B(三个根)
  证明(去重):    {bundle.dedup/1024:.0f} KB
  验证:           {t_verify*1e3:.1f} ms(vs 全量 {base_attempt*1e3:.0f} ms)
  矿工承诺开销:   +{overhead:.0%}(Merkle={HASH_NAME})
  安全:           跳过 f 工作被抓概率 1-(1-f)^{K_CHAIN};f=30% -> 99.92%""")


if __name__ == "__main__":
    main()
