#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
256MB 那档的 2.81x 到底有多结实?

NarrowNet 的赌注是"GPU 装不下足够多的链"。那么这个优势就应该是**显存的函数**——
它对 24GB 的 4090 成立,不代表对 80GB 的数据中心卡成立。这里用实测数据回答:
GPU 在大池子时是被算力卡还是被显存卡,以及换更大显存会怎样。

推演对 GPU 是保守的(用实测的"链数越多每链越慢"曲线插值,没假设线性加速)。
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "..", "narrownet_gpu_vs_cpu.json")))

print("=== GPU 每条链的吞吐:被算力卡还是被显存卡? ===")
print(f"{'池子':>7}{'链数':>7}{'总吞吐':>11}{'每链吞吐':>12}{'每链ns/步':>11}")
pts = []
for r in d["rows"]:
    g = r["gpu_exact"]
    n = g["chains"]
    per = g["msteps"] / n
    print(f"{r['pool_mb']:>5}MB{n:>7}{g['msteps']:>10.1f}M{per:>11.4f}M{g['ns_per_step']:>11.0f}")
    pts.append((n, per))

print("""
  链数越少 -> 每链越快(争用小),饱和在 ~0.17 Msteps/链。
  大池子时 GPU 的算力没跑满,是**显存装不下更多链** => 优势是显存的函数。
""")


def interp(chains, pts):
    """按实测的 链数->每链吞吐 曲线插值;链数越多每链越慢。"""
    xs = sorted(pts)
    if chains <= xs[0][0]:
        return xs[0][1]
    if chains >= xs[-1][0]:
        return xs[-1][1]
    for i in range(len(xs) - 1):
        if xs[i][0] <= chains <= xs[i + 1][0]:
            t = (chains - xs[i][0]) / (xs[i + 1][0] - xs[i][0])
            return xs[i][1] + t * (xs[i + 1][1] - xs[i][1])
    return xs[-1][1]


CARDS = [(24, "RTX 4090 24GB(实测)"), (48, "RTX 6000 Ada 48GB"),
         (80, "H100 / A100 80GB"), (141, "H200 141GB")]

for pool_mb in (256, 1024):
    row = next(r for r in d["rows"] if r["pool_mb"] == pool_mb)
    cpu = row["cpu"]["msteps"]
    cpu_note = "(内存受限只开了 %d 线程)" % row["cpu_threads"] if row["cpu"].get("ram_capped") else ""
    print(f"=== 反事实推演 @ {pool_mb}MB 池(CPU 侧固定 {cpu:.1f} Msteps/s {cpu_note})===")
    for vram, name in CARDS:
        chains = int(vram * 1024 * 0.72 // pool_mb)
        if chains < 1:
            print(f"  {name:<24} 装不下一条链")
            continue
        gm = chains * interp(chains, pts)
        ratio = cpu / gm
        print(f"  {name:<24}{chains:>5} 链 ->{gm:>7.1f} Msteps/s   CPU/GPU = {ratio:>5.2f}x   "
              f"{'✅ CPU 赢' if ratio > 1 else '❌ GPU 赢'}")
    print()

print("""结论:
  2.81x 是 **24GB 消费卡**上的数字,不是"CPU 对 GPU"的普遍结论。
  池子要选多大,取决于你把谁当对手——对手是数据中心卡,门槛就得往上抬。""")
