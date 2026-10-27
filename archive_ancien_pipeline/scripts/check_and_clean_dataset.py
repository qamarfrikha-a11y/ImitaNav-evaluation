#!/usr/bin/env python3
"""
Verification et nettoyage des datasets ImitaNav (raw, dagger, processed).

Ne SUPPRIME et n'ECRASE rien : les fichiers corriges sont ecrits a cote,
avec le suffixe _clean.

Usage :
    python3 check_and_clean_dataset.py                       # rapport seul
    python3 check_and_clean_dataset.py --fix                 # + ecrit les datasets corriges
    python3 check_and_clean_dataset.py --root ~/imitanav_evaluation --fix

Sorties (avec --fix), dans <root>/data/processed/ :
    dataset_clean.npz       dataset.npz actuel sans doublons (ordre conserve)
    dataset_all_clean.npz   raw + TOUTES les sessions dagger, sans doublons
    cleaning_report.txt     rapport complet
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np

OBS_DIM, ACT_DIM = 40, 2


def row_keys(obs, act):
    """Cle binaire par ligne (obs + action) pour detecter les doublons exacts."""
    both = np.ascontiguousarray(np.hstack([obs, act]).astype(np.float32))
    return [r.tobytes() for r in both]


def dedup(obs, act):
    """Garde la 1ere occurrence de chaque (obs, action). Retourne obs, act, indices gardes."""
    seen, keep = set(), []
    for i, k in enumerate(row_keys(obs, act)):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return obs[keep], act[keep], keep


def check_arrays(name, obs, act, log):
    """Controles de base. Retourne True si tout est OK."""
    ok = True
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        log(f"  [ERREUR] {name}: obs shape {obs.shape}, attendu (N,{OBS_DIM})")
        ok = False
    if act.ndim != 2 or act.shape[1] != ACT_DIM:
        log(f"  [ERREUR] {name}: act shape {act.shape}, attendu (N,{ACT_DIM})")
        ok = False
    if len(obs) != len(act):
        log(f"  [ERREUR] {name}: desalignement obs={len(obs)} act={len(act)}")
        ok = False
    if len(obs) == 0:
        log(f"  [ERREUR] {name}: fichier vide")
        ok = False
    if ok and (not np.isfinite(obs).all() or not np.isfinite(act).all()):
        log(f"  [ERREUR] {name}: NaN/inf detectes")
        ok = False
    return ok


def load_pairs(folder, obs_prefix, act_prefix, log):
    """Charge les paires (obs_*, act_*) d'un dossier. Retourne {timestamp: (obs, act)}."""
    pairs = {}
    obs_files = sorted(glob.glob(os.path.join(folder, f"{obs_prefix}_*.npy")))
    for f in obs_files:
        ts = os.path.basename(f)[len(obs_prefix) + 1:-4]
        af = os.path.join(folder, f"{act_prefix}_{ts}.npy")
        if not os.path.exists(af):
            log(f"  [ERREUR] {os.path.basename(f)}: fichier action manquant")
            continue
        obs, act = np.load(f), np.load(af)
        if check_arrays(os.path.basename(f), obs, act, log):
            pairs[ts] = (obs, act)
    # actions sans obs
    for af in glob.glob(os.path.join(folder, f"{act_prefix}_*.npy")):
        ts = os.path.basename(af)[len(act_prefix) + 1:-4]
        if not os.path.exists(os.path.join(folder, f"{obs_prefix}_{ts}.npy")):
            log(f"  [ERREUR] {os.path.basename(af)}: fichier obs manquant")
    return pairs


def analyse_sessions(label, pairs, log, already_seen=None):
    """Doublons internes + sessions redondantes (n'apportent aucune ligne nouvelle)."""
    seen = set(already_seen) if already_seen else set()
    redundant, hashes = [], {}
    log(f"\n== {label}: {len(pairs)} fichiers ==")
    for ts, (obs, act) in sorted(pairs.items()):
        keys = row_keys(obs, act)
        internal = len(keys) - len(set(keys))
        new = len(set(keys) - seen)
        h = hashlib.md5(obs.tobytes() + act.tobytes()).hexdigest()
        tag = []
        if internal:
            tag.append(f"{internal} doublons internes")
        if h in hashes:
            tag.append(f"IDENTIQUE a {hashes[h]}")
        hashes.setdefault(h, ts)
        if new == 0:
            tag.append("REDONDANT (0 ligne nouvelle)")
            redundant.append(ts)
        log(f"  {ts}: {len(keys):5d} pas | nouveaux={new:5d}" + (" | " + ", ".join(tag) if tag else ""))
        seen.update(keys)
    return seen, redundant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/imitanav_evaluation",
                    help="racine du projet (contient data/raw, data/dagger, data/processed)")
    ap.add_argument("--fix", action="store_true", help="ecrit les datasets corriges (_clean)")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    raw_dir = os.path.join(root, "data", "raw")
    dag_dir = os.path.join(root, "data", "dagger")
    proc_dir = os.path.join(root, "data", "processed")
    ds_path = os.path.join(proc_dir, "dataset.npz")

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    if not os.path.isdir(os.path.join(root, "data")):
        sys.exit(f"Dossier introuvable : {root}/data  (utilise --root)")

    # ---- 1. raw ----
    raw = load_pairs(raw_dir, "obs", "act", log)
    seen, raw_red = analyse_sessions("data/raw (demonstrations BC)", raw, log)
    n_raw = len(seen)

    # ---- 2. dagger ----
    dag = load_pairs(dag_dir, "dagger_obs", "dagger_act", log)
    seen_all, dag_red = analyse_sessions("data/dagger (corrections HG-DAgger)", dag, log, already_seen=seen)

    # ---- 3. dataset.npz actuel ----
    ds_clean = None
    if os.path.exists(ds_path):
        d = np.load(ds_path)
        obs, act = d["observations"], d["actions"]
        log(f"\n== data/processed/dataset.npz ==")
        if check_arrays("dataset.npz", obs, act, log):
            keys = row_keys(obs, act)
            n_unique = len(set(keys))
            extra = len(keys) - n_unique
            log(f"  lignes: {len(keys)} | uniques: {n_unique} | doublons: {extra} ({100 * extra / len(keys):.2f}%)")
            # conflits : meme obs, actions differentes
            first = {}
            conflicts = 0
            for o, a in zip(np.ascontiguousarray(obs).tolist(), act):
                k = tuple(o)
                if k in first and not np.array_equal(first[k], a):
                    conflicts += 1
                first.setdefault(k, a)
            log(f"  conflits (meme obs, actions differentes): {conflicts}")
            if extra:
                # blocs dupliques
                pos = {}
                dup_idx = []
                for i, k in enumerate(keys):
                    if k in pos:
                        dup_idx.append(i)
                    else:
                        pos[k] = i
                log(f"  doublons situes aux indices {min(dup_idx)} .. {max(dup_idx)}")
                log("  -> ATTENTION: dataset.npz contient des doublons, ne pas l'utiliser tel quel.")
            else:
                log("  -> OK, aucun doublon.")
            ds_clean = dedup(obs, act)
    else:
        log(f"\n[info] {ds_path} introuvable")

    # ---- 4. resume ----
    log("\n== RESUME ==")
    log(f"  raw uniques:            {n_raw}")
    log(f"  raw + dagger uniques:   {len(seen_all)}")
    if raw_red:
        log(f"  fichiers raw redondants:    {raw_red}")
    if dag_red:
        log(f"  fichiers dagger redondants: {dag_red}  (peuvent etre ecartes)")

    # ---- 5. ecriture ----
    if args.fix:
        os.makedirs(proc_dir, exist_ok=True)
        if ds_clean is not None:
            o, a, _ = ds_clean
            out = os.path.join(proc_dir, "dataset_clean.npz")
            np.savez(out, observations=o, actions=a)
            log(f"\n[OK] {out}  ({len(o)} pas)")
        all_o = [p[0] for p in raw.values()] + [p[0] for p in dag.values()]
        all_a = [p[1] for p in raw.values()] + [p[1] for p in dag.values()]
        if all_o:
            o, a, _ = dedup(np.concatenate(all_o), np.concatenate(all_a))
            out = os.path.join(proc_dir, "dataset_all_clean.npz")
            np.savez(out, observations=o, actions=a)
            log(f"[OK] {out}  ({len(o)} pas, raw + dagger sans doublons)")
        rep = os.path.join(proc_dir, "cleaning_report.txt")
        with open(rep, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[OK] {rep}")
    else:
        print("\n(rapport seul : relance avec --fix pour ecrire les datasets corriges)")


if __name__ == "__main__":
    main()
