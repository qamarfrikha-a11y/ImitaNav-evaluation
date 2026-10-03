#!/usr/bin/env python3
"""
Genere les graphiques PNG pour le README / rapport de stage a partir :
- des logs d'entrainement (fichiers texte contenant les lignes "Epoch X/Y | train_loss=... | val_loss=...")
- des CSV d'evaluation quantitative (results/evaluation_G*.csv)

Usage:
    python3 generate_report_charts.py

Genere dans ./report_assets/ :
    training_curve.png       - courbe train/val loss du modele multigoal
    success_rate_by_goal.png - taux de reussite par goal
    baseline_vs_dagger.png   - comparaison G2 baseline vs multigoal
    collision_timeout_breakdown.png - repartition des types d'echec par goal
"""
import os
import re
import glob
import csv
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser('~/imitanav_evaluation')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUT_DIR = os.path.join(os.getcwd(), 'report_assets')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'


def parse_training_log(log_path):
    """Extrait (epochs, train_loss, val_loss) d'un fichier de log d'entrainement."""
    epochs, train_losses, val_losses = [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss=([\d.]+)\s+\|\s+val_loss=([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
    return epochs, train_losses, val_losses


def plot_training_curve():
    """Cherche le log d'entrainement multigoal le plus recent et trace la courbe."""
    candidates = sorted(glob.glob(os.path.expanduser('~/train_multigoal_log_*.txt')))
    if not candidates:
        print("Aucun log d'entrainement multigoal trouve (train_multigoal_log_*.txt dans ~/), "
              "graphique non genere.")
        return
    log_path = candidates[-1]
    epochs, train_losses, val_losses = parse_training_log(log_path)
    if not epochs:
        print(f"Aucune ligne Epoch trouvee dans {log_path}")
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label='Train loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Entrainement du modele BC multi-goal')
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = os.path.join(OUT_DIR, 'training_curve.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Genere: {out_path}")


def load_eval_csv(csv_path):
    """Retourne (n_essais, n_reussis, n_collision, n_timeout)."""
    if not os.path.exists(csv_path):
        return None
    n, reussis, collisions, timeouts = 0, 0, 0, 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            if int(row['success']) == 1:
                reussis += 1
            if int(row.get('blocking_collision', 0)) == 1:
                collisions += 1
            if int(row.get('timeout', 0)) == 1:
                timeouts += 1
    return n, reussis, collisions, timeouts


def plot_success_rate_by_goal():
    goals = ['G1', 'G2', 'G3', 'G4', 'G5']
    rates = []
    labels_n = []
    for g in goals:
        stats = load_eval_csv(os.path.join(RESULTS_DIR, f'evaluation_{g}.csv'))
        if stats is None:
            rates.append(0)
            labels_n.append('N/A')
            continue
        n, reussis, _, _ = stats
        rates.append(100 * reussis / n if n else 0)
        labels_n.append(f'{reussis}/{n}')

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(goals, rates, color='#4C72B0')
    for bar, label in zip(bars, labels_n):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  label, ha='center', fontsize=9)
    plt.ylim(0, 110)
    plt.ylabel('Taux de reussite (%)')
    plt.title('Taux de reussite par goal — modele multi-goal (BC + HG-DAgger)')
    plt.grid(axis='y', alpha=0.3)
    out_path = os.path.join(OUT_DIR, 'success_rate_by_goal.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Genere: {out_path}")


def plot_baseline_vs_dagger():
    """Compare un goal (par defaut G2) baseline vs multigoal si les deux CSV existent."""
    goal = 'G2'
    baseline_stats = load_eval_csv(os.path.join(RESULTS_DIR, f'evaluation_{goal}_baseline.csv'))
    multigoal_stats = load_eval_csv(os.path.join(RESULTS_DIR, f'evaluation_{goal}.csv'))

    if baseline_stats is None or multigoal_stats is None:
        print(f"CSV baseline ou multigoal manquant pour {goal}, graphique non genere.")
        return

    n_b, r_b, _, _ = baseline_stats
    n_m, r_m, _, _ = multigoal_stats
    rate_b = 100 * r_b / n_b if n_b else 0
    rate_m = 100 * r_m / n_m if n_m else 0

    plt.figure(figsize=(5.5, 4.5))
    bars = plt.bar(['Sans DAgger\n(baseline)', 'Avec DAgger\n(multi-goal)'],
                    [rate_b, rate_m], color=['#C44E52', '#55A868'])
    for bar, (n, r) in zip(bars, [(n_b, r_b), (n_m, r_m)]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  f'{r}/{n}', ha='center', fontsize=10)
    plt.ylim(0, 110)
    plt.ylabel('Taux de reussite (%)')
    plt.title(f'Impact de HG-DAgger sur {goal} (jamais vu par le modele baseline)')
    plt.grid(axis='y', alpha=0.3)
    out_path = os.path.join(OUT_DIR, 'baseline_vs_dagger.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Genere: {out_path}")


def plot_collision_timeout_breakdown():
    goals = ['G1', 'G2', 'G3', 'G4', 'G5']
    reussi, collision, timeout = [], [], []
    for g in goals:
        stats = load_eval_csv(os.path.join(RESULTS_DIR, f'evaluation_{g}.csv'))
        if stats is None:
            reussi.append(0)
            collision.append(0)
            timeout.append(0)
            continue
        n, r, c, t = stats
        reussi.append(r)
        collision.append(c)
        timeout.append(t)

    plt.figure(figsize=(7, 4.5))
    width = 0.6
    plt.bar(goals, reussi, width, label='Reussi', color='#55A868')
    plt.bar(goals, collision, width, bottom=reussi, label='Collision', color='#C44E52')
    bottom2 = [r + c for r, c in zip(reussi, collision)]
    plt.bar(goals, timeout, width, bottom=bottom2, label='Timeout', color='#DD8452')
    plt.ylabel("Nombre d'essais")
    plt.title("Repartition des issues par goal (modele multi-goal)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    out_path = os.path.join(OUT_DIR, 'collision_timeout_breakdown.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Genere: {out_path}")


if __name__ == '__main__':
    plot_training_curve()
    plot_success_rate_by_goal()
    plot_baseline_vs_dagger()
    plot_collision_timeout_breakdown()
    print(f"\nTous les graphiques disponibles sont dans: {OUT_DIR}")
    print("Copie ce dossier dans ton depot (ex: docs/assets/ ou report_assets/) "
          "et reference-les dans le README avec des liens relatifs.")
