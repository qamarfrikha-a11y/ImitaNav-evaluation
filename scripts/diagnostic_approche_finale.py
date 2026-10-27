#!/usr/bin/env python3
"""
Diagnostic de l'approche finale : le robot qui tourne sur place pres du but.

Hypothese testee : dans l'approche finale (scriptee), le controleur envoie
    angulaire = 1.5 * angle_au_but   (plafonne a +-2.0)
avec un angle_au_but calcule a partir de l'ODOMETRIE. Si l'odometrie a derive,
cet angle est faux et le robot tourne indefiniment.

Methode : pour chaque pas ou |cmd_ang| < 1.9 (non ssature), on retrouve l'angle
vu par le controleur (= cmd_ang / 1.5) et on le compare a l'angle REEL calcule
avec la position et le cap reels (trajectoire enregistree).
    ecart = angle_vu_par_le_controleur - angle_reel
Un ecart proche de 0 = odometrie coherente ; un grand ecart constant = derive.

Usage :
    python3 scripts/diagnostic_approche_finale.py results/experiments/trials_g1_40.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

GOALS = {"G1": (5.5, 1.5), "G2": (6.5, -2.0), "G3": (1.0, 2.0), "G4": (6.0, 0.0), "G5": (0.5, -2.0)}
ZONE_M = 1.4          # distance reelle au but en dessous de laquelle l'approche finale est active
ANG_GAIN = 1.5


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def circ_mean(vals):
    s = sum(math.sin(v) for v in vals) / len(vals)
    c = sum(math.cos(v) for v in vals) / len(vals)
    return math.atan2(s, c), math.hypot(s, c)      # moyenne, concentration (1 = tous identiques)


def trial_offset(path, goal):
    gx, gy = goal
    offs = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                x, y, yaw = float(r["x"]), float(r["y"]), float(r["yaw"])
                ang = float(r["cmd_ang"])
            except (KeyError, ValueError):
                continue
            if math.hypot(gx - x, gy - y) > ZONE_M or abs(ang) >= 1.9:
                continue
            true_angle = wrap(math.atan2(gy - y, gx - x) - yaw)
            seen_angle = ang / ANG_GAIN
            offs.append(wrap(seen_angle - true_angle))
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--traj-dir", default=None)
    args = ap.parse_args()
    path = Path(args.csv)
    tdir = Path(args.traj_dir) if args.traj_dir else path.parent / "trajectories"
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["outcome"] in ("success", "collision", "timeout")]

    print(f"{'essai':44s} {'resultat':9s} {'dist_fin':>8s} {'pas':>4s} {'ecart moyen':>12s} {'regularite':>10s}")
    summary = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["outcome"], r["trial_uid"])):
        p = tdir / f"{r['trial_uid']}.csv"
        if not p.exists():
            continue
        offs = trial_offset(p, GOALS[r["goal"]])
        if len(offs) < 5:
            continue
        m, conc = circ_mean(offs)
        summary[r["outcome"]].append(math.degrees(m))
        if r["outcome"] == "timeout" or len(summary["success"]) <= 3 and r["outcome"] == "success":
            print(f"{r['trial_uid']:44s} {r['outcome']:9s} {float(r['final_dist_m']):8.2f} {len(offs):4d} "
                  f"{math.degrees(m):+10.0f}°  {conc:9.2f}")

    print("\n=== Resume : ecart moyen (controleur - realite), en degres ===")
    for oc in ("success", "timeout"):
        v = summary.get(oc, [])
        if v:
            a = sorted(abs(x) for x in v)
            print(f"  {oc:8s}: {len(v):2d} essais | ecart absolu median = {a[len(a) // 2]:5.0f}° "
                  f"| min {a[0]:.0f}° | max {a[-1]:.0f}°")
    print("\nLecture : succes ~ 0° et timeouts pres du but a plusieurs dizaines de degres, avec une")
    print("regularite proche de 1, = le controleur vise une direction fausse (derive d'odometrie).")


if __name__ == "__main__":
    main()