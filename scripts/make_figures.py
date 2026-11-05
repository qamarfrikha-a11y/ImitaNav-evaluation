#!/usr/bin/env python3
"""Figures des resultats (generees depuis les CSV) -> media/images/."""
import csv, math, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "media" / "images"
GOALS = ["G1", "G2", "G3", "G4", "G5"]


def load(pattern):
    rows = []
    for f in sorted(EXP.glob(pattern)):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def sel(rows, goal=None, seed=None, model=None):
    return [r for r in rows
            if (goal is None or r["goal"] == goal)
            and (seed is None or r["train_seed"] == str(seed))
            and (model is None or r["model_id"] == model)]


def kn(rows):
    return sum(int(r["success"]) for r in rows), len(rows)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


A = load("trials_g[1-5]_dagall.csv")
B = load("trials_g[1-5]_dag10.csv")
G1 = load("trials_g1_v2.csv")
C5, C10, CB = "#4C78A8", "#E4572E", "#7F7F7F"

# ---- Figure 1 : G1, BC contre HG-DAgger, par seed ----
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.38
for i, (m, lab, col) in enumerate((("bc", "BC", CB), ("bc_dag_g1", "BC + HG-DAgger", C5))):
    vals = [100 * kn(sel(G1, seed=s, model=m))[0] / max(1, kn(sel(G1, seed=s, model=m))[1]) for s in (0, 1, 2)]
    tot = kn(sel(G1, model=m))
    xs = [s + (i - 0.5) * w for s in (0, 1, 2)]
    ax.bar(xs, vals, w, label=f"{lab} (total {tot[0]}/{tot[1]})", color=col)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["seed 0", "seed 1", "seed 2"])
ax.set_ylabel("Succès (%) sur 10 essais"); ax.set_ylim(0, 125)
ax.set_title("G1 : BC contre BC + HG-DAgger, par seed d'entraînement")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "g1_bc_vs_hgdagger_par_seed.png", dpi=150); plt.close(fig)

# ---- Figure 2 : generalisation, 5 % contre 10 %, avec IC95 ----
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (R, lab, col) in enumerate(((A, "5 % (501 pas G2-G5)", C5), (B, "10 % (1035 pas G2-G5)", C10))):
    vals, lo, hi = [], [], []
    for g in GOALS:
        k, n = kn(sel(R, goal=g))
        a, b = wilson(k, n)
        vals.append(100 * k / n); lo.append(100 * (k / n - a)); hi.append(100 * (b - k / n))
    xs = [j + (i - 0.5) * w for j in range(5)]
    ax.bar(xs, vals, w, yerr=[lo, hi], capsize=3, label=lab, color=col)
    for x, v, h in zip(xs, vals, hi):
        ax.text(x, v + h + 1.5, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(GOALS)
ax.set_ylabel("Succès (%) sur 30 essais"); ax.set_ylim(0, 108)
ax.set_title("Généralisation : succès par goal (barres d'erreur : IC 95 %)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "generalisation_5_vs_10.png", dpi=150); plt.close(fig)

# ---- Figure 3 : par seed, 5 % contre 10 % ----
fig, axes = plt.subplots(1, 5, figsize=(11, 3.4), sharey=True)
for ax, g in zip(axes, GOALS):
    for i, (R, lab, col) in enumerate(((A, "5 %", C5), (B, "10 %", C10))):
        vals = [kn(sel(R, goal=g, seed=s))[0] for s in (0, 1, 2)]
        xs = [s + (i - 0.5) * 0.38 for s in (0, 1, 2)]
        ax.bar(xs, vals, 0.38, label=lab, color=col)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.2, str(v), ha="center", fontsize=7)
    ax.set_title(g); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["s0", "s1", "s2"])
    ax.set_ylim(0, 14)
axes[0].set_ylabel("Succès sur 10 essais")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Succès par seed d'entraînement : 5 % contre 10 %", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "generalisation_par_seed.png", dpi=150); plt.close(fig)

print("Figures ecrites dans", OUT)
for f in ("g1_bc_vs_hgdagger_par_seed.png", "generalisation_5_vs_10.png", "generalisation_par_seed.png"):
    print(" ", f, (OUT / f).stat().st_size, "octets")
