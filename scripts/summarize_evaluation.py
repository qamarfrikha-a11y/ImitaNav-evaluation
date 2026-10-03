#!/usr/bin/env python3
"""
Resume les resultats de l'evaluation quantitative (results/evaluation.csv)
en statistiques agregees, pretes a inserer dans le rapport de stage.
"""

import csv
import os
import statistics

import sys
RESULTS_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/imitanav_evaluation/results/evaluation.csv')


def main():
    if not os.path.isfile(RESULTS_CSV):
        print(f"Aucun resultat trouve dans {RESULTS_CSV}")
        return

    with open(RESULTS_CSV, newline='') as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        print("Le fichier de resultats est vide.")
        return

    n = len(rows)
    n_success = sum(int(r['success']) for r in rows)
    n_collision = sum(int(r['blocking_collision']) for r in rows)
    n_timeout = sum(int(r['timeout']) for r in rows)

    times = [float(r['time_s']) for r in rows]
    lengths = [float(r['trajectory_length_m']) for r in rows]
    smoothness = [float(r['angular_std']) for r in rows]

    times_success = [float(r['time_s']) for r in rows if int(r['success']) == 1]
    lengths_success = [float(r['trajectory_length_m']) for r in rows if int(r['success']) == 1]

    print("=== Resume de l'evaluation quantitative ===\n")
    print(f"Nombre d'essais              : {n}")
    print(f"Taux de reussite             : {n_success}/{n} ({100 * n_success / n:.1f}%)")
    print(f"Essais avec collision         : {n_collision}/{n} ({100 * n_collision / n:.1f}%)")
    print(f"Essais en timeout             : {n_timeout}/{n} ({100 * n_timeout / n:.1f}%)")
    print()
    print(f"Temps moyen (tous essais)     : {statistics.mean(times):.2f} s (ecart-type {statistics.pstdev(times):.2f})")
    if times_success:
        print(f"Temps moyen (essais reussis)  : {statistics.mean(times_success):.2f} s")
    print(f"Longueur moyenne (tous)       : {statistics.mean(lengths):.2f} m")
    if lengths_success:
        print(f"Longueur moyenne (reussis)    : {statistics.mean(lengths_success):.2f} m")
    print(f"Fluidite moyenne (std angular): {statistics.mean(smoothness):.3f} rad/s")
    print()
    print("Detail par essai :")
    for r in rows:
        status = "REUSSI" if int(r['success']) else ("COLLISION" if int(r['blocking_collision']) else "TIMEOUT")
        print(f"  #{r['trial_id']:>2}: {status:<10} temps={r['time_s']:>6}s  "
              f"traj={r['trajectory_length_m']:>6}m  std_ang={r['angular_std']}")


if __name__ == '__main__':
    main()
