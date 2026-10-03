#!/usr/bin/env python3
"""
Entraine le MLP BC sur un jeu propre de data/training_sets/.
Architecture et hyperparametres IDENTIQUES a train_bc_multigoal.py
(MLP 40-128-64-32-2, MSE, Adam 1e-3, batch 64, 150 epochs, patience 20).
A figer AVANT les essais Gazebo : ne plus rien changer apres.

Usage :
    python3 scripts/train_bc_clean.py bc_dag_g1 --seed 42
    python3 scripts/train_bc_clean.py bc_dag_g1 --seed 0 1 2    # plusieurs seeds
Sortie : models/clean/<jeu>_seed<seed>.pt et .json (log)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "data" / "training_sets"
MODELS = ROOT / "models" / "clean"

EPOCHS, BATCH, LR, PATIENCE = 150, 64, 1e-3, 20


class BCPolicy(nn.Module):
    def __init__(self, input_dim=40, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train(name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = np.load(SETS / f"{name}.npz")
    obs, act, isv = d["obs"], d["act"], d["is_val"]
    tr = TensorDataset(torch.tensor(obs[~isv]), torch.tensor(act[~isv]))
    va = TensorDataset(torch.tensor(obs[isv]), torch.tensor(act[isv]))
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g)
    va_loader = DataLoader(va, batch_size=BATCH, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{name}_seed{seed}.pt"
    best, wait, history = float("inf"), 0, []
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(x)
        tl /= len(tr)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                vl += crit(model(x), y).item() * len(x)
        vl /= len(va)
        history.append((ep, tl, vl))
        if vl < best:
            best, wait = vl, 0
            torch.save(model.state_dict(), path)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        if ep % 10 == 0 or ep == 1:
            print(f"[{name} s{seed}] epoch {ep:3d} train={tl:.5f} val={vl:.5f}")

    log = dict(set=name, seed=seed, n_train=len(tr), n_val=len(va),
               best_val_loss=best, epochs_run=len(history),
               hyperparams=dict(epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE),
               history=history)
    with open(MODELS / f"{name}_seed{seed}.json", "w") as f:
        json.dump(log, f)
    print(f"[{name} s{seed}] best_val={best:.5f} -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--seed", type=int, nargs="+", default=[42])
    a = p.parse_args()
    for s in a.seed:
        train(a.name, s)
