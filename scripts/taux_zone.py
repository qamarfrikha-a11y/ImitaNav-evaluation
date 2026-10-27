#!/usr/bin/env python3
"""
Combien d'essais atteignent la ZONE du but (distance reelle <= 1.5 m) ?

C'est la mesure de ce que fait la politique APPRISE seule : l'approche finale
(scriptee) ne prend la main qu'a partir de 1.5 m. Cette mesure ne depend donc ni
de l'approche finale, ni de l'odometrie.

Usage :
    python3 scripts/taux_zone.py results/experiments/trials_g1_40.csv
    python3 scripts/taux_zone.py results/experiments/trials_g1_40.csv --zone 2.0
    python3 scripts/taux_zone.py results/experiments/trials_g1_40.csv --max-seed 30
        (--max-seed N : garde seulement les essais trial_seed < N)
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

GOALS = {"G1": (5.5, 1.5), "G2": (6.5, -2.0), "G3": (1.0, 2.0), "G4": (6.0, 0.0), "G5": (0.5, -2.0)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def min_dist(path, goal):
    gx, gy = goal
    best = float("inf")
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                best = min(best, math.hypot(gx - float(r["x"]), gy - float(r["y"])))
            except (KeyError, ValueError):
                pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--zone", type=float, default=1.5)
    ap.add_argument("--max-seed", type=int, default=None)
    ap.add_argument("--traj-dir", default=None)
    args = ap.parse_args()
    path = Path(args.csv)
    tdir = Path(args.traj_dir) if args.traj_dir else path.parent / "trajectories"
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["outcome"] in ("success", "collision", "timeout")]
    if args.max_seed is not None:
        rows = [r for r in rows if int(r["trial_seed"]) < args.max_seed]

    data = defaultdict(list)    # methode -> [(atteint, succes)]
    seeds = defaultdict(lambda: defaultdict(list))
    missing = 0
    for r in rows:
        p = tdir / f"{r['trial_uid']}.csv"
        if not p.exists():
            missing += 1
            continue
        d = min_dist(p, GOALS[r["goal"]])
        reached = d <= args.zone
        data[r["method"]].append((reached, r["outcome"] == "success"))
        seeds[r["method"]][r["train_seed"]].append(reached)
    if missing:
        print(f"({missing} trajectoires introuvables, ignorees)")

    print(f"\n=== Essais qui atteignent la zone (distance reelle <= {args.zone} m a un moment) ===")
    print(f"{'methode':26s} {'n':>3s} {'zone':>5s} {'taux':>6s} {'IC 95 %':>14s}   {'succes':>6s} {'dont succes parmi la zone':>26s}")
    for m in sorted(data):
        n = len(data[m])
        k = sum(1 for a, _ in data[m] if a)
        s = sum(1 for _, ok in data[m] if ok)
        s_in = sum(1 for a, ok in data[m] if a and ok)
        lo, hi = wilson(k, n)
        print(f"{m:26s} {n:3d} {k:5d} {100 * k / n:5.0f}% [{100 * lo:3.0f}% - {100 * hi:3.0f}%]   {s:6d} {s_in:>12d}/{k}")

    print("\n=== Par methode et seed d'entrainement (zone atteinte) ===")
    for m in sorted(seeds):
        print(m)
        for sd in sorted(seeds[m]):
            v = seeds[m][sd]
            print(f"  seed {sd}: {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
