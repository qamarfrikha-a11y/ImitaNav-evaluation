#!/usr/bin/env python3
"""
Genere le plan d'essais de la campagne : results/experiments/plan.csv
(ou plan_pilot.csv avec --pilot). Le plan est FIGE : meme ordre (melange avec
une seed fixe) et memes poses de depart a chaque execution.

Appariement : pour un meme trial_seed, la pose de depart est identique pour
toutes les methodes et tous les objectifs ; la seed d'entrainement du modele
est train_seed = k % 3.

Experiences :
  g1_main         bc, bc_dag_g1              sur G1       30 essais chacun
  generalization  bc, bc_dag_all             sur G2..G5   20 essais chacun
  dagger_ratio    bc_dag_g1_3pct / _5pct     sur G1       20 essais chacun
  (0 % de correction = les essais k < 20 de bc dans g1_main, memes seeds)
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "training_sets" / "manifest.json"
OUT_DIR = ROOT / "results" / "experiments"

GOALS = {"G1": (5.5, 1.5), "G2": (6.5, -2.0), "G3": (1.0, 2.0),
         "G4": (6.0, 0.0), "G5": (0.5, -2.0)}
SHUFFLE_SEED = 12345
XY_NOISE = 0.10     # m
YAW_NOISE = 0.15    # rad (~8.6 deg)
N_TRAIN_SEEDS = 3

FIELDS = ["trial_uid", "experiment", "goal", "goal_x", "goal_y", "method",
          "model_id", "pct_correction", "train_seed", "trial_seed",
          "start_x", "start_y", "start_yaw", "model_path"]


def start_pose(trial_seed):
    rng = np.random.default_rng(1000 + trial_seed)
    return (round(float(rng.uniform(-XY_NOISE, XY_NOISE)), 3),
            round(float(rng.uniform(-XY_NOISE, XY_NOISE)), 3),
            round(float(rng.uniform(-YAW_NOISE, YAW_NOISE)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    sets = json.load(open(MANIFEST))["sets"]
    rows = []

    def add(exp, model, goal, k, train_seed=None):
        ts = k % N_TRAIN_SEEDS if train_seed is None else train_seed
        sx, sy, syaw = start_pose(k)
        gx, gy = GOALS[goal]
        rows.append(dict(
            trial_uid=f"{exp}__{model}__{goal}__t{k:02d}",
            experiment=exp, goal=goal, goal_x=gx, goal_y=gy,
            method="BC" if model == "bc" else "BC+HG-DAgger",
            model_id=model, pct_correction=sets[model]["pct_correction"],
            train_seed=ts, trial_seed=k, start_x=sx, start_y=sy, start_yaw=syaw,
            model_path=f"models/clean/{model}_seed{ts}.pt"))

    if args.pilot:
        for model in ("bc", "bc_dag_g1"):
            for k in range(5):
                add("pilot", model, "G1", k, train_seed=0)
        name = "plan_pilot.csv"
    else:
        for model in ("bc", "bc_dag_g1"):
            for k in range(30):
                add("g1_main", model, "G1", k)
        for goal in ("G2", "G3", "G4", "G5"):
            for model in ("bc", "bc_dag_all"):
                for k in range(20):
                    add("generalization", model, goal, k)
        for model in ("bc_dag_g1_3pct", "bc_dag_g1_5pct"):
            for k in range(20):
                add("dagger_ratio", model, "G1", k)
        name = "plan.csv"

    random.Random(SHUFFLE_SEED).shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} essais -> {OUT_DIR / name}")


if __name__ == "__main__":
    main()
