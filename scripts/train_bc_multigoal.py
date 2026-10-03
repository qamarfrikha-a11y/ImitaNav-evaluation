#!/usr/bin/env python3
"""
Entrainement du modele BC multi-goal : fusionne le dataset original (G1)
avec les corrections HG-DAgger collectees pour G2, G3, G4, G5, puis
reentraine le reseau FROM SCRATCH (pas de fine-tuning) sur l'ensemble.

Pourquoi from scratch plutot que fine-tuning ici : le modele existant a
ete entraine uniquement sur G1 et a developpe un biais fort vers ce seul
point. Repartir de ses poids risquerait de transferer ce biais. Un
entrainement propre sur donnees equilibrees donne un signal plus net sur
la vraie capacite de generalisation multi-goal.

Seed fixee pour reproductibilite (comparaison rigoureuse entre versions).
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
DAGGER_DIR = os.path.expanduser('~/imitanav_evaluation/data/dagger')
MODEL_DIR = os.path.expanduser('~/imitanav_evaluation/models')
MODEL_PATH = os.path.join(MODEL_DIR, f'bc_model_multigoal_seed{SEED}.pt')

# Fichiers dagger valides, un par nouveau goal (verifies manuellement :
# trajectoire de distance decroissante et coherente, sans saut de goal)
DAGGER_FILES = {
    'G2 (6.5,-2.0)': '20260822_130626',
    'G3 (1.0,2.0)': '20260824_182639',
    'G4 (6.0,0.0)': '20260825_005025',
    'G5 (0.5,-2.0)': '20260825_005823',
}

INPUT_DIM = 40
OUTPUT_DIM = 2
BATCH_SIZE = 64
EPOCHS = 150   # un peu plus que l'original (100) car dataset plus varie
LR = 1e-3
VAL_SPLIT = 0.15
PATIENCE = 20   # early stopping


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


def load_merged_dataset():
    data = np.load(DATASET_PATH)
    obs_list = [data['observations']]
    act_list = [data['actions']]
    n_g1 = data['observations'].shape[0]
    print(f"G1 (dataset original): {n_g1} pas")

    total_new = 0
    counts = {'G1': n_g1}
    for goal_name, ts in DAGGER_FILES.items():
        obs_path = os.path.join(DAGGER_DIR, f'dagger_obs_{ts}.npy')
        act_path = os.path.join(DAGGER_DIR, f'dagger_act_{ts}.npy')
        if not os.path.exists(obs_path):
            print(f"  [ATTENTION] fichier introuvable, ignore: {obs_path}")
            continue
        o = np.load(obs_path)
        a = np.load(act_path)
        obs_list.append(o)
        act_list.append(a)
        total_new += o.shape[0]
        counts[goal_name] = o.shape[0]
        print(f"{goal_name}: {o.shape[0]} pas")

    obs = np.concatenate(obs_list, axis=0).astype(np.float32)
    act = np.concatenate(act_list, axis=0).astype(np.float32)

    total = obs.shape[0]
    print(f"\nTotal fusionne: {total} pas")
    print(f"Repartition: " + ", ".join(f"{k}={v} ({100*v/total:.1f}%)" for k, v in counts.items()))

    if total_new / total < 0.05:
        print(f"\n  [ATTENTION] Les nouveaux goals (G2-G5) representent seulement "
              f"{100*total_new/total:.1f}% du dataset total.")
        print("  Le modele risque de rester fortement biaise vers G1.")
        print("  Envisage de collecter davantage de demonstrations/corrections")
        print("  pour G2-G5 si les resultats d'evaluation sont decevants.\n")

    return obs, act


def main():
    obs, act = load_merged_dataset()

    dataset = DemoDataset(obs, act)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0
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
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping a l'epoch {epoch} "
                      f"(pas d'amelioration depuis {PATIENCE} epochs)")
                break

    print(f"\nMeilleur val_loss: {best_val_loss:.5f}")
    print(f"Modele sauvegarde dans: {MODEL_PATH}")
    print(f"\nProchaine etape recommandee : evaluer sur les 5 goals avec")
    print(f"  export BC_MODEL_PATH={MODEL_PATH}")
    print(f"  cd ~/imitanav_evaluation/scripts")
    print(f"  ./run_evaluation.sh 10 G1   (repeter pour G2 G3 G4 G5)")


if __name__ == '__main__':
    main()
