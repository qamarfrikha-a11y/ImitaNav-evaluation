#!/usr/bin/env python3
"""
Fusionne le dataset BC initial avec les corrections DAgger, et sauvegarde un
nouveau dataset pret pour train_bc.py.

Usage :
    python3 merge_dagger_data.py            # ne fusionne QUE la derniere session DAgger
    python3 merge_dagger_data.py --all      # fusionne TOUTES les sessions dans data/dagger/

Par defaut, seule la session la plus recente (dagger_obs_*.npy / dagger_act_*.npy
avec le timestamp le plus eleve) est fusionnee -- utile si des sessions
precedentes contenaient des corrections erronees que tu ne veux pas reutiliser.
Ces anciens fichiers restent sur le disque (rien n'est supprime), ils sont
juste ignores tant que --all n'est pas precise.

Cycle DAgger complet recommande :
    1. python3 merge_dagger_data.py        (ce script, derniere session seulement)
    2. python3 scripts/train_bc.py         (reentrainement sur dataset enrichi)
    3. Relancer dagger_session_node avec le nouveau bc_model.pt
    4. Corriger les erreurs restantes, Ctrl+C pour sauvegarder
    5. Retour a l'etape 1
"""

import argparse
import glob
import os
import shutil
import time

import numpy as np

HOME = os.path.expanduser("~")
PROCESSED_DATASET = os.path.join(HOME, "imitanav_evaluation/data/processed/dataset.npz")
DAGGER_DIR = os.path.join(HOME, "imitanav_evaluation/data/dagger")
OUTPUT_DATASET = PROCESSED_DATASET  # ecrase le dataset traite -- une sauvegarde est faite avant
BACKUP_DIR = os.path.join(HOME, "imitanav_evaluation/data/processed/backups")


def load_original_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset original introuvable : {path}\n"
            "Lance d'abord scripts/build_dataset.py pour le creer."
        )
    data = np.load(path)
    obs, act = data["observations"], data["actions"]
    print(f"Dataset original charge : {obs.shape[0]} pas | obs={obs.shape} act={act.shape}")
    return obs, act


def load_dagger_corrections(dagger_dir, latest_only=True):
    obs_files = sorted(glob.glob(os.path.join(dagger_dir, "dagger_obs_*.npy")))
    act_files = sorted(glob.glob(os.path.join(dagger_dir, "dagger_act_*.npy")))

    if len(obs_files) != len(act_files):
        raise RuntimeError(
            f"Nombre de fichiers obs ({len(obs_files)}) et act ({len(act_files)}) "
            "different dans data/dagger/ -- verifie qu'aucune sauvegarde n'a echoue."
        )

    if not obs_files:
        print("Aucune correction DAgger trouvee dans data/dagger/. Rien a fusionner.")
        return None, None

    if latest_only:
        obs_files = [obs_files[-1]]
        act_files = [act_files[-1]]
        print(f"Mode --latest (par defaut) : une seule session prise en compte "
              f"({len(glob.glob(os.path.join(dagger_dir, 'dagger_obs_*.npy'))) - 1} "
              f"session(s) plus ancienne(s) ignoree(s))")
    else:
        print(f"Mode --all : {len(obs_files)} session(s) prises en compte")

    all_obs, all_act = [], []
    total_steps = 0
    for obs_f, act_f in zip(obs_files, act_files):
        obs = np.load(obs_f)
        act = np.load(act_f)
        if obs.shape[0] != act.shape[0]:
            raise RuntimeError(
                f"Desalignement dans {os.path.basename(obs_f)} : "
                f"{obs.shape[0]} obs vs {act.shape[0]} act. Fichier ignore."
            )
        total_steps += obs.shape[0]
        all_obs.append(obs)
        all_act.append(act)
        print(f"  {os.path.basename(obs_f)}: {obs.shape[0]} pas de correction")

    print(f"\nTotal corrections DAgger fusionnees : {total_steps} pas sur {len(obs_files)} session(s)")
    return np.concatenate(all_obs, axis=0), np.concatenate(all_act, axis=0)


def backup_dataset(path):
    if not os.path.exists(path):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"dataset_{timestamp}.npz")
    shutil.copy2(path, backup_path)
    print(f"Sauvegarde de l'ancien dataset -> {backup_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true",
        help="Fusionne toutes les sessions DAgger au lieu de seulement la derniere"
    )
    args = parser.parse_args()

    print("=== Fusion dataset BC + corrections DAgger ===\n")

    orig_obs, orig_act = load_original_dataset(PROCESSED_DATASET)

    print("\nRecherche des corrections DAgger...")
    dagger_obs, dagger_act = load_dagger_corrections(DAGGER_DIR, latest_only=not args.all)

    if dagger_obs is None:
        print("\nDataset inchange, aucun reentrainement necessaire pour l'instant.")
        return

    if dagger_obs.shape[1] != orig_obs.shape[1]:
        raise RuntimeError(
            f"Dimension d'observation incoherente : "
            f"original={orig_obs.shape[1]}, dagger={dagger_obs.shape[1]}. "
            "Verifie que build_observation() est identique entre le script de "
            "collecte initial et dagger_session_node.py."
        )

    merged_obs = np.concatenate([orig_obs, dagger_obs], axis=0)
    merged_act = np.concatenate([orig_act, dagger_act], axis=0)

    print(f"\nDataset fusionne : {merged_obs.shape[0]} pas au total "
          f"({orig_obs.shape[0]} initiaux + {dagger_obs.shape[0]} corrections DAgger)")

    backup_dataset(OUTPUT_DATASET)

    np.savez(OUTPUT_DATASET, observations=merged_obs, actions=merged_act)
    print(f"\nNouveau dataset sauvegarde dans : {OUTPUT_DATASET}")
    print("\nProchaine etape : python3 scripts/train_bc.py")


if __name__ == "__main__":
    main()
