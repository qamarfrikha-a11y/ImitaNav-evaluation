#!/usr/bin/env python3
"""
Construit des jeux d'entrainement PROPRES (sans doublon) depuis data/raw et
data/dagger, sans jamais lire data/processed/dataset.npz.

Sorties : data/training_sets/<nom>.npz + manifest.json
Chaque .npz contient : obs, act, is_val, source, group
  source : 0 = demonstration BC (G1), 1 = correction HG-DAgger G1,
           2 = correction HG-DAgger G2..G5

Jeux produits :
  bc, bc_dag_g1, bc_dag_g1_x2 (variante : corrections G1 dupliquees),
  bc_dag_all, bc_dag_g1_3pct, bc_dag_g1_5pct,
  bc_dag_g1_unfiltered (variante : avec les 8 sessions exclues)

Usage : python3 scripts/build_training_sets.py
"""
import glob
import json
import math
import os
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DAG = ROOT / "data" / "dagger"
OUT = ROOT / "data" / "training_sets"

SEED = 42
BLOCK = 50       # taille des blocs pour la separation train/val
VAL_FRAC = 0.15

# Sessions DAgger des nouveaux goals (d'apres train_bc_multigoal.py).
# Toutes les autres sessions sont attribuees a G1 (deduction : le goal
# n'est pas enregistre dans les .npy -- a confirmer).
OTHER_GOALS = {
    "G2": "20260822_130626",
    "G3": "20260824_182639",
    "G4": "20260825_005025",
    "G5": "20260825_005823",
}

# Sessions dont la distance au goal est SATUREE (9 a 12 m pendant toute la
# session) : incompatibles avec G1 (a 5.7 m du depart), entree "goal"
# incoherente. Exclues de tous les jeux sauf bc_dag_g1_unfiltered.
SUSPECT_SESSIONS = {
    "20260723_110925", "20260724_235818", "20260725_015641", "20260725_020439",
    "20260725_020955", "20260725_024735", "20260725_031722", "20260725_032111",
}


def ts_of(path, prefix):
    return os.path.basename(path)[len(prefix):-len(".npy")]


def load_groups():
    bc, g1, other, suspect = [], [], [], []
    for f in sorted(glob.glob(str(RAW / "obs_*.npy"))):
        ts = ts_of(f, "obs_")
        o = np.load(f)
        a = np.load(RAW / f"act_{ts}.npy")
        assert len(o) == len(a), f"taille obs/act differente : {ts}"
        bc.append(dict(name=f"bc_{ts}", obs=o, act=a, source=0))
    other_ts = set(OTHER_GOALS.values())
    for f in sorted(glob.glob(str(DAG / "dagger_obs_*.npy"))):
        ts = ts_of(f, "dagger_obs_")
        o = np.load(f)
        a = np.load(DAG / f"dagger_act_{ts}.npy")
        assert len(o) == len(a), f"taille obs/act differente : {ts}"
        if ts in other_ts:
            other.append(dict(name=f"dag_other_{ts}", obs=o, act=a, source=2))
        elif ts in SUSPECT_SESSIONS:
            suspect.append(dict(name=f"dag_suspect_{ts}", obs=o, act=a, source=1))
        else:
            g1.append(dict(name=f"dag_g1_{ts}", obs=o, act=a, source=1))
    return bc, g1, other, suspect


def take_prefix(groups, n):
    """Prend les n premiers pas en parcourant les groupes DANS L'ORDRE DONNE
    (ici : chronologique). Les pas restent contigus au sein d'une session ;
    deux paliers sont emboites (le palier 5 % contient le palier 3 %)."""
    out, remaining = [], n
    for g in groups:
        if remaining <= 0:
            break
        k = min(remaining, len(g["obs"]))
        out.append(dict(g, obs=g["obs"][:k], act=g["act"][:k]))
        remaining -= k
    assert remaining == 0, "pas assez de corrections pour ce palier"
    return out


def take_share(groups, n):
    """Prend n pas au total, repartis proportionnellement entre les sessions
    (chaque session : ses premiers pas, dans l'ordre chronologique).
    Emboite : un n plus grand contient un n plus petit."""
    total = sum(len(g["obs"]) for g in groups)
    ks = [int(round(n * len(g["obs"]) / total)) for g in groups]
    ks[ks.index(max(ks))] += n - sum(ks)
    return [dict(g, obs=g["obs"][:k], act=g["act"][:k])
            for g, k in zip(groups, ks) if k > 0]


def val_mask(name, n):
    """Separation par blocs de BLOCK pas, deterministe par groupe :
    evite la fuite entre pas voisins (quasi identiques)."""
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
    n_blocks = math.ceil(n / BLOCK)
    block_is_val = rng.random(n_blocks) < VAL_FRAC
    return np.repeat(block_is_val, BLOCK)[:n]


def assemble(groups, dup_g1_corr=False):
    obs, act, isv, src, grp = [], [], [], [], []
    for g in groups:
        n = len(g["obs"])
        obs.append(g["obs"]); act.append(g["act"])
        isv.append(val_mask(g["name"], n))
        src.append(np.full(n, g["source"]))
        grp.extend([g["name"]] * n)
    obs = np.concatenate(obs).astype(np.float32)
    act = np.concatenate(act).astype(np.float32)
    isv = np.concatenate(isv)
    src = np.concatenate(src)
    grp = np.array(grp)
    if dup_g1_corr:  # duplique seulement les pas d'ENTRAINEMENT (pas de fuite)
        d = (src == 1) & (~isv)
        obs = np.concatenate([obs, obs[d]]); act = np.concatenate([act, act[d]])
        isv = np.concatenate([isv, isv[d]]); src = np.concatenate([src, src[d]])
        grp = np.concatenate([grp, grp[d]])
    return obs, act, isv, src, grp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bc, g1, other, suspect = load_groups()
    n_bc = sum(len(g["obs"]) for g in bc)
    n_g1 = sum(len(g["obs"]) for g in g1)
    n_ot = sum(len(g["obs"]) for g in other)
    n_sus = sum(len(g["obs"]) for g in suspect)
    print(f"BC={n_bc}  corrections G1 valides={n_g1}  "
          f"sessions exclues={n_sus}  corrections G2-G5={n_ot}")
    assert (n_bc, n_g1, n_sus, n_ot) == (6004, 3151, 1836, 501), \
        "totaux inattendus (attendu 6004 / 3151 / 1836 / 501) : verifier les fichiers"

    # Ordre CHRONOLOGIQUE des sessions G1 : les premieres corrections ont ete
    # collectees avec le modele le moins entraine (logique HG-DAgger iterative).
    g1_chrono = sorted(g1, key=lambda g: g["name"])

    def n_for(p):  # p = n / (n_bc + n)
        return int(round(p * n_bc / (1 - p)))

    def n_for_new(p):  # p = n / (n_bc + n_g1 + n)
        return int(round(p * (n_bc + n_g1) / (1 - p)))

    sets = {
        "bc": (bc, False),
        "bc_dag_g1": (bc + g1, False),
        "bc_dag_g1_x2": (bc + g1, True),
        "bc_dag_g1_unfiltered": (bc + g1 + suspect, False),
        "bc_dag_all": (bc + g1 + other, False),
        # Lecture A : paliers de corrections G1 (BC + x % de corrections G1)
        "bc_dag_g1_3pct": (bc + take_prefix(g1_chrono, n_for(0.03)), False),
        "bc_dag_g1_5pct": (bc + take_prefix(g1_chrono, n_for(0.05)), False),
        # Lecture B : paliers de corrections des NOUVEAUX goals (G2-G5) ajoutees
        # a BC + corrections G1. 0 % = bc_dag_g1 ; ~5 % = bc_dag_all (toutes).
        "bc_dag_g1_new3pct": (bc + g1 + take_share(other, n_for_new(0.03)), False),
    }

    manifest = {"seed": SEED, "block": BLOCK, "val_frac": VAL_FRAC,
                "pct_formula": "n_corr / (n_bc + n_corr)", "sets": {}}
    for name, (groups, dup) in sets.items():
        obs, act, isv, src, grp = assemble(groups, dup)
        np.savez(OUT / f"{name}.npz", obs=obs, act=act, is_val=isv,
                 source=src, group=grp)
        n = len(obs)
        n_corr = int(((src == 1) | (src == 2)).sum())
        manifest["sets"][name] = dict(
            n_total=n, n_bc=int((src == 0).sum()),
            n_corr_g1=int((src == 1).sum()), n_corr_other=int((src == 2).sum()),
            pct_correction=round(100 * n_corr / n, 2),
            pct_new_goal_corr=round(100 * int((src == 2).sum()) / n, 2),
            n_train=int((~isv).sum()), n_val=int(isv.sum()),
            groups=sorted(set(grp.tolist())))
        s = manifest["sets"][name]
        print(f"{name:16s} total={n:6d} bc={s['n_bc']} g1={s['n_corr_g1']} "
              f"autres={s['n_corr_other']} corr={s['pct_correction']}% "
              f"new={s['pct_new_goal_corr']}% "
              f"train={s['n_train']} val={s['n_val']}")
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nEcrit dans {OUT}")


if __name__ == "__main__":
    main()
