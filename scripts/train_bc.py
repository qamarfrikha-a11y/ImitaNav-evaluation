#!/usr/bin/env python3
"""
Entrainement du modele de Behavioral Cloning (BC) sur le dataset collecte.
Architecture: MLP simple. Entree = 40 (36 LiDAR + goal_dist + goal_angle + 2 dernieres actions).
Sortie = 2 (vitesse lineaire, vitesse angulaire).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATASET_PATH = os.path.expanduser('~/imitanav_evaluation/data/processed/dataset.npz')
MODEL_DIR = os.path.expanduser('~/imitanav_evaluation/models')
MODEL_PATH = os.path.join(MODEL_DIR, f'bc_model_seed{SEED}.pt')

INPUT_DIM = 40
OUTPUT_DIM = 2
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
VAL_SPLIT = 0.15


class DemoDataset(Dataset):
    def __init__(self, obs, act):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.act = torch.tensor(act, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


class BCPolicy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    data = np.load(DATASET_PATH)
    obs, act = data['observations'], data['actions']
    print(f"Dataset charge: {obs.shape[0]} pas | obs={obs.shape} act={act.shape}")

    dataset = DemoDataset(obs, act)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_set)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nMeilleur val_loss: {best_val_loss:.5f}")
    print(f"Modele sauvegarde dans: {MODEL_PATH}")


if __name__ == '__main__':
    main()
