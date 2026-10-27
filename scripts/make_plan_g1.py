#!/usr/bin/env python3
"""
Plan d'essais restreint : goal G1 uniquement.
Par defaut : modeles bc et bc_dag_g1. Reutilise make_plan.py SANS le modifier
(memes poses de depart, memes seeds, meme format de ligne).

Usage :
    python3 scripts/make_plan_g1.py              # 40 essais PAR methode (80 au total)
    python3 scripts/make_plan_g1.py --n 20       # 20 par methode (40 au total)
    python3 scripts/make_plan_g1.py --n 20 --models bc_dag_g1_x2 bc_dag_g1_unfiltered
        (autres jeux d'entrainement, memes poses de depart ; "method" = nom du jeu)

Sortie : results/experiments/plan_g1_<n>.csv  (plan_g1_ablation_<n>.csv avec --models)
"""
import argparse
import csv
import json
import random

import make_plan as mp   # meme dossier : scripts/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="essais par methode")
    ap.add_argument("--models", nargs="+", default=["bc", "bc_dag_g1"],
                    help="jeux de training_sets a evaluer (par defaut : bc bc_dag_g1)")
    args = ap.parse_args()
    labels = {"bc": "BC", "bc_dag_g1": "BC+HG-DAgger"}

    sets = json.load(open(mp.MANIFEST))["sets"]
    custom = args.models != ["bc", "bc_dag_g1"]
    for m in args.models:
        if m not in sets:
            raise SystemExit(f"jeu inconnu : {m} (disponibles : {sorted(sets)})")
    exp = "g1_ablation" if custom else "g1_main"
    rows = []
    for model in args.models:
        for k in range(args.n):
            ts = k % mp.N_TRAIN_SEEDS
            sx, sy, syaw = mp.start_pose(k)
            gx, gy = mp.GOALS["G1"]
            rows.append(dict(
                trial_uid=f"{exp}__{model}__G1__t{k:02d}",
                experiment=exp, goal="G1", goal_x=gx, goal_y=gy,
                method=labels.get(model, model),
                model_id=model, pct_correction=sets[model]["pct_correction"],
                train_seed=ts, trial_seed=k, start_x=sx, start_y=sy, start_yaw=syaw,
                model_path=f"models/clean/{model}_seed{ts}.pt"))

    random.Random(mp.SHUFFLE_SEED).shuffle(rows)   # ordre melange, fige
    mp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = mp.OUT_DIR / (f"plan_g1_ablation_{args.n}.csv" if custom else f"plan_g1_{args.n}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mp.FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    seeds = sorted({r["model_path"] for r in rows})
    print(f"{len(rows)} essais ({args.n} par methode, {len(args.models)} jeux) -> {out}")
    print("Modeles necessaires :")
    for s in seeds:
        print("  ", s)


if __name__ == "__main__":
    main()
