# Plots for docs/llm_tlb_miss_3c_analysis.ko.md section 9, from the JSON written by tlb_set_histogram.py:
#   <prefix>_hist.png          whole-layer L2 TLB reference share per set (all chiplets)
#   <prefix>_touched_hist.png  distribution of distinct sets touched per window of W L2 refs (chiplet 0)
#   <prefix>_zoom.png          64 consecutive windows from the heaviest GEMM of each layer: which sets are live
# Usage: plot_tlb_set_hist.py <dir with deepseek_b4_<step>_<layer>.json> <prefix>
import json, os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

D, prefix = sys.argv[1], sys.argv[2]
STEPS = [("prefill", "Prefill"), ("decode1", "Decode step 1"), ("decode2", "Decode step 2")]
LAYERS = [("layer0", "layer 0"), ("layer1", "layer 1")]
SURFACE, INK, INK2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
RAMP = LinearSegmentedColormap.from_list("blue_seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
data = {(s, l): json.load(open(os.path.join(D, f"deepseek_b4_{s}_{l}.json"))) for s, _ in STEPS for l, _ in LAYERS}


def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def touched(d): return [sum(1 for v in w[1:] if v) for w in d["windows"]]


# 1) whole-layer set histogram
fig, axes = plt.subplots(3, 2, figsize=(11, 8.6), sharex=True, sharey=True)
for r, (s, stitle) in enumerate(STEPS):
    for c, (l, ltitle) in enumerate(LAYERS):
        ax = axes[r][c]; d = data[(s, l)]; tot = sum(d["hist"]); share = [100 * v / tot for v in d["hist"]]
        ax.bar(range(d["sets"]), share, width=0.72, color=BAR, linewidth=0)
        ax.axhline(100 / d["sets"], color=INK2, linewidth=0.8); clean(ax); ax.set_ylim(0, 1.25)
        ax.set_title(f"{stitle}, {ltitle}", loc="left", fontsize=10, color=INK)
        ax.text(0.02, 0.985, f"L2 refs {tot:,} · even sets {sum(share[0::2]):.1f}% · min–max {min(share):.2f}–{max(share):.2f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK2)
        if r == 2: ax.set_xlabel("L2 TLB set (vpn % 128)")
        if c == 0: ax.set_ylabel("share of L2 TLB refs (%)")
axes[0][1].text(127, 100 / 128 + 0.02, "uniform 0.78%", fontsize=7.5, color=INK2, va="bottom", ha="right")
fig.suptitle("DeepSeek b4s256 — L2 TLB references by set over the WHOLE layer (all chiplets): every set is used about equally",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(f"{prefix}_hist.png", dpi=150); plt.close(fig)

# 2) distinct sets touched per window
fig, axes = plt.subplots(3, 2, figsize=(11, 8.6), sharex=True, sharey=True)
for r, (s, stitle) in enumerate(STEPS):
    for c, (l, ltitle) in enumerate(LAYERS):
        ax = axes[r][c]; d = data[(s, l)]; t = touched(d); n = len(t)
        counts = [0] * 129
        for v in t: counts[v] += 1
        ax.bar(range(129), [100 * v / n for v in counts], width=0.9, color=BAR, linewidth=0); clean(ax)
        med = statistics.median(t)
        ax.set_title(f"{stitle}, {ltitle}", loc="left", fontsize=10, color=INK)
        ax.text(0.99, 0.97, f"windows {n:,} · median {med:.0f} sets · ≤ 32 sets: {sum(1 for v in t if v <= 32) / n * 100:.0f}% of windows",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        if r == 2: ax.set_xlabel(f"distinct L2 TLB sets touched in a window of {d['window']} L2 refs (of 128)")
        if c == 0: ax.set_ylabel("share of windows (%)")
        ax.set_xticks([0, 8, 16, 32, 48, 64, 96, 128])
fig.suptitle("DeepSeek b4s256 — how many of the 128 L2 TLB sets are live AT THE SAME TIME (chiplet 0, trace-driven model)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(f"{prefix}_touched_hist.png", dpi=150); plt.close(fig)

# 3) zoom: 64 consecutive windows from the middle of the heaviest kernel of each layer
fig = plt.figure(figsize=(11, 9.6))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.06], hspace=0.55, wspace=0.18)
vmax = 0.0; panels = []
for r, (s, stitle) in enumerate(STEPS):
    for c, (l, ltitle) in enumerate(LAYERS):
        d = data[(s, l)]; win = d["windows"]
        per_k = {}
        for i, w in enumerate(win): per_k.setdefault(w[0], []).append(i)
        kid = max(per_k, key=lambda k: len(per_k[k])); idx = per_k[kid]
        mid = idx[len(idx) // 2]; sel = win[max(idx[0], mid - 32):max(idx[0], mid - 32) + 64]
        mat = [[v / d["window"] for v in w[1:]] for w in sel]; vmax = max(vmax, max(max(row) for row in mat))
        kname = next(k["name"] for k in d["kernels"] if k["kernel"] == kid)
        panels.append((r, c, stitle, ltitle, kid, kname, mat))
for r, c, stitle, ltitle, kid, kname, mat in panels:
    ax = fig.add_subplot(gs[r, c])
    im = ax.imshow(mat, aspect="auto", cmap=RAMP, vmin=0, vmax=vmax, interpolation="nearest")
    t = [sum(1 for v in row if v) for row in mat]
    ax.set_title(f"{stitle}, {ltitle} · kernel {kid} GEMM · median {statistics.median(t):.0f} sets live",
                 loc="left", fontsize=9, color=INK)
    ax.set_xlabel("L2 TLB set (vpn % 128)"); ax.set_ylabel("window (time →)")
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 127])
    for sp in ax.spines.values(): sp.set_visible(False)
cax = fig.add_subplot(gs[3, :])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
cb.set_label(f"share of the window's L2 TLB refs that go to the set (white = set not used in that window)", color=INK2)
fig.suptitle("DeepSeek b4s256 — zoom into the heaviest GEMM of each layer: each row is one window of 1,024 L2 refs",
             x=0.01, ha="left", fontsize=11)
fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.07)
fig.savefig(f"{prefix}_zoom.png", dpi=150); plt.close(fig)

for s, stitle in STEPS:
    for l, ltitle in LAYERS:
        d = data[(s, l)]; tot = sum(d["hist"]); share = [100 * v / tot for v in d["hist"]]; t = sorted(touched(d))
        print(f"{stitle:14s} {ltitle}: L2 refs {tot:>11,}  even {sum(share[0::2]):5.1f}%  share min-max {min(share):.2f}-{max(share):.2f}%  "
              f"sets/window median {statistics.median(t):.0f} (p10 {t[len(t)//10]}, p90 {t[9*len(t)//10]}), <=32: {sum(1 for v in t if v <= 32)/len(t)*100:.0f}%  windows {len(t)}")
