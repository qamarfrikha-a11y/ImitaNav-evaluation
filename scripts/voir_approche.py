#!/usr/bin/env python3
"""
Montre ce que fait le robot quand il est pres du but (dans les 1.4 m), pas a pas.

Usage :
    python3 scripts/voir_approche.py g1_main__bc_dag_g1__G1__t22
    python3 scripts/voir_approche.py g1_main__bc_dag_g1__G1__t18 --every 1.0 --max-lignes 60

Colonnes : temps (s) | x, y | distance reelle au but | angle reel au but (deg) |
           cmd lineaire | cmd angulaire | contacts
"""
import argparse
import csv
import math
from pathlib import Path

GOALS = {"G1": (5.5, 1.5), "G2": (6.5, -2.0), "G3": (1.0, 2.0), "G4": (6.0, 0.0), "G5": (0.5, -2.0)}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uid")
    ap.add_argument("--traj-dir", default="results/experiments/trajectories")
    ap.add_argument("--goal", default="G1")
    ap.add_argument("--zone", type=float, default=1.4)
    ap.add_argument("--every", type=float, default=0.5, help="pas d'echantillonnage en secondes")
    ap.add_argument("--max-lignes", type=int, default=50)
    args = ap.parse_args()

    gx, gy = GOALS[args.goal]
    path = Path(args.traj_dir) / f"{args.uid}.csv"
    rows = list(csv.DictReader(open(path, newline="")))
    print(f"{path.name} : {len(rows)} pas ; but {args.goal} = ({gx}, {gy}) ; zone < {args.zone} m\n")
    print(f"{'t(s)':>7s} {'x':>6s} {'y':>6s} {'dist':>5s} {'angle_reel':>10s} {'lin':>5s} {'ang':>6s} {'bump':>4s}")
    last_t, shown, entered = -1e9, 0, None
    for r in rows:
        t, x, y, yaw = float(r["t_s"]), float(r["x"]), float(r["y"]), float(r["yaw"])
        d = math.hypot(gx - x, gy - y)
        if d > args.zone:
            continue
        if entered is None:
            entered = t
            print(f"--- entree dans la zone a t = {t:.1f} s ---")
        if t - last_t < args.every:
            continue
        last_t = t
        a = math.degrees(wrap(math.atan2(gy - y, gx - x) - yaw))
        print(f"{t:7.1f} {x:6.2f} {y:6.2f} {d:5.2f} {a:+9.0f}° {float(r['cmd_lin']):5.2f} "
              f"{float(r['cmd_ang']):+6.2f} {r['bump_events']:>4s}")
        shown += 1
        if shown >= args.max_lignes:
            print("... (limite atteinte, utilise --max-lignes pour en voir plus)")
            break
    if entered is None:
        print("Le robot n'est jamais entre dans la zone.")


if __name__ == "__main__":
    main()