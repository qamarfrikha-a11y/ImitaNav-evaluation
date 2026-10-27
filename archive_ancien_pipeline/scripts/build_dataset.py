#!/usr/bin/env python3
"""
Fusionne toutes les demonstrations individuelles (data/raw/obs_*.npy, act_*.npy)
en un seul dataset pret pour l'entrainement (data/processed/dataset.npz).
"""

import glob
import os
import numpy as np

RAW_DIR = os.path.expanduser('~/imitanav_evaluation/data/raw')
PROCESSED_DIR = os.path.expanduser('~/imitanav_evaluation/data/processed')

def main():
    obs_files = sorted(glob.glob(os.path.join(RAW_DIR, 'obs_*.npy')))
    act_files = sorted(glob.glob(os.path.join(RAW_DIR, 'act_*.npy')))

    if len(obs_files) == 0:
        print("Aucune demonstration trouvee dans", RAW_DIR)
        return

    assert len(obs_files) == len(act_files), "Nombre de fichiers obs/act different !"

    all_obs = []
    all_act = []
    for obs_f, act_f in zip(obs_files, act_files):
        obs = np.load(obs_f)
        act = np.load(act_f)
        assert obs.shape[0] == act.shape[0], f"Desalignement dans {obs_f}"
        all_obs.append(obs)
        all_act.append(act)
        print(f"{os.path.basename(obs_f)}: {obs.shape[0]} pas")

    X = np.concatenate(all_obs, axis=0)
    y = np.concatenate(all_act, axis=0)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, 'dataset.npz')
    np.savez(out_path, observations=X, actions=y)

    print(f"\nDataset final: {X.shape[0]} pas au total")
    print(f"Observations: {X.shape}, Actions: {y.shape}")
    print(f"Sauvegarde dans: {out_path}")

if __name__ == '__main__':
    main()
