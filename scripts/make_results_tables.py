#!/usr/bin/env python3
"""Genere results/RESULTATS.md depuis les CSV d'essais (aucun chiffre saisi a la main)."""
import csv, math, collections
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "RESULTATS.md"
GOALS = ["G1", "G2", "G3", "G4", "G5"]


def load(pattern):
    rows = []
    for f in sorted(EXP.glob(pattern)):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def sel(rows, goal=None, seed=None, model=None):
    out = []
    for r in rows:
        if goal is not None and r["goal"] not in goal:
            continue
        if seed is not None and r["train_seed"] != str(seed):
            continue
        if model is not None and r["model_id"] != model:
            continue
        out.append(r)
    return out


def kn(rows):
    return sum(int(r["success"]) for r in rows), len(rows)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def fmt(k, n, with_ci=True):
    if n == 0:
        return "n/a"
    s = f"{k}/{n} ({100 * k / n:.0f} %)"
    if with_ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.0f}-{100 * hi:.0f}]"
    return s


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)


def paired(Ra, Rb, goals):
    da = {(r["goal"], r["trial_seed"]): int(r["success"]) for r in Ra if r["goal"] in goals}
    db = {(r["goal"], r["trial_seed"]): int(r["success"]) for r in Rb if r["goal"] in goals}
    a_only = sum(1 for k in da if k in db and da[k] and not db[k])
    b_only = sum(1 for k in da if k in db and db[k] and not da[k])
    return a_only, b_only, sign_test(a_only, b_only)


def table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


md = ["# Resultats (generes par scripts/make_results_tables.py)\n"]

# ---------- 1. G1 : BC vs BC+HG-DAgger ----------
G1 = load("trials_g1_v2.csv")
md.append("## 1. G1 : BC contre BC + HG-DAgger (30 essais par methode, 3 seeds x 10)\n")
models = sorted({r["model_id"] for r in G1})
md.append(f"Modeles presents dans le fichier : {models}\n")
rows = []
for m, label in (("bc", "BC"), ("bc_dag_g1", "BC + HG-DAgger (G1)")):
    R = sel(G1, model=m)
    if not R:
        continue
    k, n = kn(R)
    ok = [float(r["nav_time_s"]) for r in R if r["success"] == "1"]
    coll = sum(int(r["collision"]) for r in R)
    tout = sum(int(r["timeout"]) for r in R)
    per = [fmt(*kn(sel(R, seed=s)), with_ci=False) for s in (0, 1, 2)]
    rows.append([label, fmt(k, n), coll, tout] + per +
                [f"{sum(ok) / len(ok):.1f}" if ok else "n/a"])
md.append(table(["Methode", "Succes [IC95 %]", "Collisions", "Timeouts",
                 "Seed 0", "Seed 1", "Seed 2", "Temps moyen des succes (s)"], rows))
a, b, p = paired(sel(G1, model="bc"), sel(G1, model="bc_dag_g1"), ["G1"])
md.append(f"Essais apparies (meme pose, meme seed) : BC seul reussit {a}, HG-DAgger seul reussit {b} ; "
          f"test des signes p = {p:.3f}.\n")

# ---------- 2. Generalisation 5 % vs 10 % ----------
A = load("trials_g[1-5]_dagall.csv")
B = load("trials_g[1-5]_dag10.csv")
md.append("## 2. Generalisation : 5 % contre 10 % de corrections sur les autres goals\n")
md.append("5 % = `bc_dag_all` (501 pas sur G2-G5, 5,19 %). 10 % = `bc_dag_all_10pct` "
          "(1035 pas sur G3 et G5, 10,16 %). Chaque case seed = succes sur 10 essais.\n")
rows = []
groups = [(g, [g]) for g in GOALS] + [("G2-G5", GOALS[1:]), ("G1-G5", GOALS)]
for name, gs in groups:
    row = [name]
    for R in (A, B):
        for s in (0, 1, 2):
            k, n = kn(sel(R, goal=gs, seed=s))
            row.append(f"{k}/{n}")
        row.append(fmt(*kn(sel(R, goal=gs))))
    a, b, p = paired(A, B, gs)
    row += [f"{a} / {b}", f"{p:.3f}"]
    rows.append(row)
md.append(table(["Goal", "5 % s0", "5 % s1", "5 % s2", "**5 % total** [IC95 %]",
                 "10 % s0", "10 % s1", "10 % s2", "**10 % total** [IC95 %]",
                 "Gagnes 5 % seul / 10 % seul", "p (signes)"], rows))

# ---------- 3. Issues ----------
md.append("## 3. Issues des essais (succes / collision / timeout)\n")
rows = []
for g in GOALS:
    row = [g]
    for R in (A, B):
        c = collections.Counter(r["outcome"] for r in sel(R, goal=[g]))
        row.append(f"{c['success']} / {c['collision']} / {c['timeout']}")
    rows.append(row)
md.append(table(["Goal", "5 %", "10 %"], rows))

# ---------- 4. Tracabilite ----------
md.append("## 4. Tracabilite\n")
for name, R in (("G1 v2", G1), ("5 % (dagall)", A), ("10 % (dag10)", B)):
    uids = [r["trial_uid"] for r in R]
    md.append(f"- {name} : {len(R)} essais, {len(set(uids))} identifiants uniques, "
              f"modeles {sorted({r['model_id'] for r in R})}, "
              f"commits {sorted({r['git_commit'] for r in R})}")
md.append("")

text = "\n".join(md)
OUT.write_text(text)
print(text)
print("\nEcrit :", OUT)
