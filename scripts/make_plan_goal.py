#!/usr/bin/env python3
"""
Plan d'essais pour les goals G2 a G5 (ou tout goal defini dans make_plan.GOALS).

Meme logique que make_plan_g1.py : reutilise make_plan.py SANS le modifier, donc
- memes poses de depart : le trial_seed k donne toujours la meme pose (comme sur G1),
- memes seeds d'entrainement : train_seed = k % N_TRAIN_SEEDS,
- essais apparies : chaque methode est evaluee sur les memes poses.

Usage :
    python3 scripts/make_plan_goal.py                      # G2..G5, 30 essais par methode
    python3 scripts/make_plan_goal.py --goals G2 --n 10    # test rapide sur G2
    python3 scripts/make_plan_goal.py --models bc bc_dag_all bc_dag_g1   # avec un controle
    python3 scripts/make_plan_goal.py --goals G1 G2 G3 G4 G5 --models bc_dag_all \
        --exp dagall_main --tag dagall                  # BC+HG-DAgger seul, 5 goals

Sortie : results/experiments/plan_<goal>_<n>[_<tag>].csv   (un fichier par goal)
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import make_plan as mp   # meme dossier : scripts/

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "bc": "BC",
    "bc_dag_all": "BC+HG-DAgger",
    "bc_dag_g1": "BC+HG-DAgger (G1 seul)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", nargs="+", default=["G2", "G3", "G4", "G5"],
                    help="goals a evaluer (par defaut : G2 G3 G4 G5)")
    ap.add_argument("--n", type=int, default=30, help="essais par methode et par goal")
    ap.add_argument("--models", nargs="+", default=["bc", "bc_dag_all"],
                    help="jeux de training_sets a comparer (par defaut : bc bc_dag_all)")
    ap.add_argument("--exp", default="goals_main", help="nom de l'experience")
    ap.add_argument("--tag", default="",
                    help="suffixe du nom des plans (evite d'ecraser un plan existant)")
    args = ap.parse_args()

    sets = json.load(open(mp.MANIFEST))["sets"]
    for m in args.models:
        if m not in sets:
            raise SystemExit(f"jeu inconnu : {m} (disponibles : {sorted(sets)})")
    for g in args.goals:
        if g not in mp.GOALS:
            raise SystemExit(f"goal inconnu : {g} (disponibles : {sorted(mp.GOALS)})")

    mp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = set()

    for goal in args.goals:
        gx, gy = mp.GOALS[goal]
        rows = []
        for model in args.models:
            for k in range(args.n):
                ts = k % mp.N_TRAIN_SEEDS
                sx, sy, syaw = mp.start_pose(k)
                model_path = f"models/clean/{model}_seed{ts}.pt"
                if not (ROOT / model_path).is_file():
                    missing.add(model_path)
                rows.append(dict(
                    trial_uid=f"{args.exp}__{model}__{goal}__t{k:02d}",
                    experiment=args.exp, goal=goal, goal_x=gx, goal_y=gy,
                    method=LABELS.get(model, model),
                    model_id=model, pct_correction=sets[model]["pct_correction"],
                    train_seed=ts, trial_seed=k,
                    start_x=sx, start_y=sy, start_yaw=syaw,
                    model_path=model_path))

        random.Random(mp.SHUFFLE_SEED).shuffle(rows)   # ordre melange, fige
        suffix = f"_{args.tag}" if args.tag else ""
        out = mp.OUT_DIR / f"plan_{goal.lower()}_{args.n}{suffix}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=mp.FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"{goal} ({gx}, {gy}) : {len(rows)} essais "
              f"({args.n} par methode, {len(args.models)} methodes) -> {out}")

    if missing:
        print("\nATTENTION : modeles introuvables (les essais correspondants echoueront) :")
        for p in sorted(missing):
            print("  ", p)
        sys.exit(1)
    print("\nTous les modeles necessaires sont presents.")


if __name__ == "__main__":
    main()