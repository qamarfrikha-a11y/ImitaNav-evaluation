#!/usr/bin/env python3
"""
OU les robots echouent ? Carte des trajectoires par methode + resume par zone.
Utilise les trajectoires deja enregistrees (aucune nouvelle simulation).

Usage :
    python3 scripts/carte_echecs.py results/experiments/trials_g1_40.csv
    python3 scripts/carte_echecs.py results/experiments/trials_g1_all.csv --max-seed 20

Sorties :
    results/experiments/carte_echecs.png   (si matplotlib est installe)
    resume par zone dans le terminal
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

# Monde (obstacles.world) : (centre x, centre y, largeur selon x, largeur selon y)
WALLS = [(3, -2.5, 8, 0.2), (3, 2.5, 8, 0.2), (-1, 0, 0.2, 5), (7, -1, 0.2, 3),
         (7, 1.5, 0.2, 2), (4, 0.5, 0.2, 4)]
BOXES = [(2, 1, 0.5, 0.5), (5.5, -1.5, 0.6, 0.6)]
CYL = [(1, -1.2, 0.3)]
GOAL = (5.5, 1.5)

# Zones du parcours : depart -> cylindre -> passage (x=4, y de -2.5 a -1.5) -> cote but
ZONES = [
    ("avant le passage (x < 3.3)",          lambda x, y: x < 3.3),
    ("dans / pres du passage (3.3<=x<=4.7, y<-1)", lambda x, y: 3.3 <= x <= 4.7 and y < -1.0),
    ("apres le passage, cote obstacle (x>4.7, y<0)", lambda x, y: x > 4.7 and y < 0.0),
    ("cote du but (y>=0, x>3.3)",           lambda x, y: x > 3.3 and y >= 0.0),
]


def zone(x, y):
    for name, f in ZONES:
        if f(x, y):
            return name
    return "autre"


def read_traj(path):
    xs, ys = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                xs.append(float(r["x"]))
                ys.append(float(r["y"]))
            except (KeyError, ValueError):
                pass
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--traj-dir", default=None, help="dossier des trajectoires (defaut : a cote du CSV)")
    ap.add_argument("--max-seed", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None, help="garde seulement ces methodes")
    args = ap.parse_args()

    path = Path(args.csv)
    tdir = Path(args.traj_dir) if args.traj_dir else path.parent / "trajectories"
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["outcome"] in ("success", "collision", "timeout")]
    if args.max_seed is not None:
        rows = [r for r in rows if int(r["trial_seed"]) < args.max_seed]
    if args.only:
        rows = [r for r in rows if r["method"] in args.only]

    methods = sorted({r["method"] for r in rows})
    data = defaultdict(list)          # methode -> [(row, xs, ys)]
    missing = 0
    for r in rows:
        p = tdir / f"{r['trial_uid']}.csv"
        if not p.exists():
            missing += 1
            continue
        xs, ys = read_traj(p)
        if xs:
            data[r["method"]].append((r, xs, ys))
    if missing:
        print(f"({missing} trajectoires introuvables dans {tdir}, ignorees)")

    # --- resume par zone : ou finissent les essais rates ---
    print("\n=== Ou finissent les essais rates (derniere position connue) ===")
    for m in methods:
        n = len(data[m])
        c = Counter()
        for r, xs, ys in data[m]:
            if r["outcome"] != "success":
                c[(r["outcome"], zone(xs[-1], ys[-1]))] += 1
        print(f"\n{m} ({n} essais avec trajectoire)")
        for oc in ("collision", "timeout"):
            tot = sum(v for (o, _), v in c.items() if o == oc)
            print(f"  {oc}: {tot}")
            for (o, z), v in sorted(c.items(), key=lambda kv: -kv[1]):
                if o == oc:
                    print(f"      {v:3d}  {z}")

    # --- carte ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        print("\nmatplotlib absent : pas de carte (pip install matplotlib).")
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 6.2), squeeze=False)
    colors = {"success": "#2e9e44", "collision": "#d62728", "timeout": "#ff9f1c"}
    for ax, m in zip(axes[0], methods):
        for cx, cy, w, h in WALLS:
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, color="#555"))
        for cx, cy, w, h in BOXES:
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, color="#999"))
        for cx, cy, rad in CYL:
            ax.add_patch(Circle((cx, cy), rad, color="#999"))
        ax.add_patch(Circle(GOAL, 0.5, fill=False, ls="--", color="k"))
        ax.plot(*GOAL, "k*", ms=12)
        cnt = Counter()
        for r, xs, ys in data[m]:
            oc = r["outcome"]
            cnt[oc] += 1
            ax.plot(xs, ys, color=colors[oc], lw=0.8, alpha=0.45)
            if oc != "success":
                ax.plot(xs[-1], ys[-1], "x", color=colors[oc], ms=7)
        ax.set_xlim(-1.5, 7.5)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{m}\nsucces {cnt['success']} | collisions {cnt['collision']} | timeouts {cnt['timeout']}",
                     fontsize=10)
    fig.suptitle("Trajectoires par methode (vert = succes, rouge = collision, orange = timeout ; x = position finale)")
    fig.tight_layout()
    out = path.parent / "carte_echecs.png"
    fig.savefig(out, dpi=130)
    print(f"\nCarte : {out}")


if __name__ == "__main__":
    main()