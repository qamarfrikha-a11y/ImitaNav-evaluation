#!/usr/bin/env python3
"""Met a jour la section Resultats de README.md (entre marqueurs) depuis results/RESULTATS.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RES = ROOT / "results" / "RESULTATS.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

res = RES.read_text()
i, j = res.index("## 1."), res.index("## 4.")
lines = [("#" + l) if l.startswith("## ") else l for l in res[i:j].splitlines()]
tables = "\n".join(lines)

section = f"""{START}
## Résultats : généralisation aux goals G2 à G5 (paliers 5 % et 10 %)

Évaluation en simulation Gazebo (arène avec obstacles). Politique : MLP 40-128-64-32-2 entraîné par
clonage de comportement (BC), puis amélioré par HG-DAgger (corrections humaines pendant l'exécution).
30 essais par modèle et par goal (10 par seed d'entraînement 0, 1, 2), mêmes poses de départ pour tous
les modèles (essais appariés), filtre de sécurité actif. Les intervalles entre crochets sont des
intervalles de confiance à 95 % (Wilson) ; s0, s1, s2 désignent les seeds d'entraînement ; « p (signes) »
est un test des signes sur les essais appariés.

| Jeu d'entraînement | Pas BC | Corrections G1 | Corrections G2-G5 | Part G2-G5 |
|---|---|---|---|---|
| `bc` | 6004 | 0 | 0 | 0 % |
| `bc_dag_g1` | 6004 | 3151 | 0 | 0 % |
| `bc_dag_all` (« 5 % ») | 6004 | 3151 | 501 | 5,19 % |
| `bc_dag_all_10pct` (« 10 % ») | 6004 | 3151 | 1035 | 10,16 % |

Corrections G2-G5 par goal : G2 176 → 176, G3 74 → 364, G4 197 → 197, G5 54 → 298 (palier 5 % → 10 %).

### Ce qu'il faut retenir

- **Généralisation, G3** : le palier 10 % fait passer le succès de 6/30 (20 %) à 25/30 (83 %). Sur les essais
  appariés, 19 sont gagnés et aucun perdu, et le gain est présent pour les trois seeds (3→8, 0→7, 3→10). C'est le goal qui a reçu le plus de
  corrections (74 → 364 pas).
- **G2, G4, G5** : pas d'effet mesurable. **G1** : 80 % → 63 %, non significatif (p = 0,30).
- **La variance entre seeds est du même ordre que les effets** : par exemple sur G4, la seed 0 passe de 10/10 à
  3/10 et la seed 2 de 0/10 à 10/10, sans nouvelle collecte sur G4 entre les deux paliers.

{tables}
### Analyse des échecs sur G5

Les échecs ont été localisés à partir des positions finales (trajectoires enregistrées) :

- **Seed 2** : les 10 échecs, aux deux paliers, finissent contre `cylinder_obstacle_1` (1,0 ; -1,2), placé
  sur la ligne vers le goal. Sans filtre de sécurité le robot le heurte (essai diagnostic) ; avec le filtre il reste
  bloqué (plus de 1000 déclenchements du filtre par essai). La seed 0 contourne l'obstacle par la gauche (10/10).
- **Seed 1** : les 7 collisions au cylindre du palier 5 % disparaissent au palier 10 %, mais 7 autres échecs apparaissent
  ailleurs dans l'arène (4 vers x ≈ 4,3 ; y ≈ 1,7-2,2 et 3 à l'est, x ≥ 5,8) : le problème se déplace.
- Le taux de succès global de G5 (43 % → 37 %) masque donc un changement de mode d'échec.

### Limites de la campagne de généralisation

- **Peu d'essais et peu de modèles** : 30 essais par cellule (IC95 % d'environ ±17 points) mais seulement
  3 modèles par palier. Les essais d'un même modèle ne sont pas indépendants : les p-valeurs du tableau (test des signes
  sur les essais) sont donc indicatives et trop optimistes. Seul l'effet sur G3, visible pour les trois seeds, résiste à ce bruit.
- **Trois seeds, un entraînement par seed** : la variance d'entraînement est du même ordre que les effets mesurés
  (le protocole est reproductible : le modèle 5 % seed 0 a été réentraîné à l'identique).
- **Quantité et répartition confondues** : le palier 5 % est réparti sur G2-G5, le palier 10 % est ciblé sur G3 et G5.
- **Un seul tour de HG-DAgger par palier** : les corrections ont été collectées sur le modèle 5 % puis ajoutées en une fois.
- **Collecte du palier 10 %** : deux sessions G5 très courtes (29 et 23 pas), conditions de simulation légèrement
  différentes (dock présent lors des premières sessions) et modèle de la première session G3 non confirmé.
- **Validation** : l'ensemble de validation passe de 1570 à 1599 pas entre les deux paliers.
- **Versions du code** : G1 testé au commit 73fd5e3, palier 5 % au commit 0b1ca12, palier 10 % au commit 228a398.
- **Étiquetage des goals de collecte** : le goal des sessions G1 est déduit (non enregistré dans les `.npy`),
  vérifié par les journaux ROS pour une session ; les nouvelles sessions l'enregistrent dans un JSON.
- **Pas de BC sur G2-G5 à 30 essais** : l'apport du HG-DAgger par rapport à BC sur les nouveaux goals n'est pas
  évalué dans cette campagne (la première campagne, 6 essais, n'est pas comparable).
- **4 essais redémarrés** après échec de démarrage de la simulation (G4 t05, G5 t05, t18, t25).

### Reproduire la campagne de généralisation

```bash
python3 scripts/build_training_sets.py
python3 scripts/train_bc_clean.py bc_dag_all_10pct --seed 0 1 2
N=30 TAG=dag10 ./scripts/run_goals.sh G1 G2 G3 G4 G5
python3 scripts/make_results_tables.py
python3 scripts/update_readme.py
```
{END}
"""

text = README.read_text() if README.exists() else ""
if START in text and END in text:
    a, b = text.index(START), text.index(END) + len(END)
    text = text[:a] + section.strip() + text[b:]
else:
    text = text.rstrip() + "\n\n" + section
README.write_text(text)
print("README mis à jour :", README)
