# Resultats (generes par scripts/make_results_tables.py)

## 1. G1 : BC contre BC + HG-DAgger (30 essais par methode, 3 seeds x 10)

Modeles presents dans le fichier : ['bc', 'bc_dag_g1']

| Methode | Succes [IC95 %] | Collisions | Timeouts | Seed 0 | Seed 1 | Seed 2 | Temps moyen des succes (s) |
|---|---|---|---|---|---|---|---|
| BC | 17/30 (57 %) [39-73] | 13 | 0 | 8/10 (80 %) | 9/10 (90 %) | 0/10 (0 %) | 29.6 |
| BC + HG-DAgger (G1) | 20/30 (67 %) [49-81] | 6 | 4 | 6/10 (60 %) | 9/10 (90 %) | 5/10 (50 %) | 42.1 |

Essais apparies (meme pose, meme seed) : BC seul reussit 5, HG-DAgger seul reussit 8 ; test des signes p = 0.581.

## 2. Generalisation : 5 % contre 10 % de corrections sur les autres goals

5 % = `bc_dag_all` (501 pas sur G2-G5, 5,19 %). 10 % = `bc_dag_all_10pct` (1035 pas sur G3 et G5, 10,16 %). Chaque case seed = succes sur 10 essais.

| Goal | 5 % s0 | 5 % s1 | 5 % s2 | **5 % total** [IC95 %] | 10 % s0 | 10 % s1 | 10 % s2 | **10 % total** [IC95 %] | Gagnes 5 % seul / 10 % seul | p (signes) |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 9/10 | 10/10 | 5/10 | 24/30 (80 %) [63-90] | 7/10 | 6/10 | 6/10 | 19/30 (63 %) [46-78] | 10 / 5 | 0.302 |
| G2 | 9/10 | 4/10 | 6/10 | 19/30 (63 %) [46-78] | 9/10 | 0/10 | 7/10 | 16/30 (53 %) [36-70] | 6 / 3 | 0.508 |
| G3 | 3/10 | 0/10 | 3/10 | 6/30 (20 %) [10-37] | 8/10 | 7/10 | 10/10 | 25/30 (83 %) [66-93] | 0 / 19 | 0.000 |
| G4 | 10/10 | 5/10 | 0/10 | 15/30 (50 %) [33-67] | 3/10 | 2/10 | 10/10 | 15/30 (50 %) [33-67] | 11 / 11 | 1.000 |
| G5 | 10/10 | 3/10 | 0/10 | 13/30 (43 %) [27-61] | 10/10 | 1/10 | 0/10 | 11/30 (37 %) [22-54] | 3 / 1 | 0.625 |
| G2-G5 | 32/40 | 12/40 | 9/40 | 53/120 (44 %) [36-53] | 30/40 | 10/40 | 27/40 | 67/120 (56 %) [47-64] | 20 / 34 | 0.076 |
| G1-G5 | 41/50 | 22/50 | 14/50 | 77/150 (51 %) [43-59] | 37/50 | 16/50 | 33/50 | 86/150 (57 %) [49-65] | 30 / 39 | 0.336 |

## 3. Issues des essais (succes / collision / timeout)

| Goal | 5 % | 10 % |
|---|---|---|
| G1 | 24 / 6 / 0 | 19 / 9 / 2 |
| G2 | 19 / 10 / 1 | 16 / 13 / 1 |
| G3 | 6 / 22 / 2 | 25 / 2 / 3 |
| G4 | 15 / 15 / 0 | 15 / 15 / 0 |
| G5 | 13 / 9 / 8 | 11 / 10 / 9 |

## 4. Tracabilite

- G1 v2 : 60 essais, 60 identifiants uniques, modeles ['bc', 'bc_dag_g1'], commits ['73fd5e3']
- 5 % (dagall) : 150 essais, 150 identifiants uniques, modeles ['bc_dag_all'], commits ['0b1ca12']
- 10 % (dag10) : 150 essais, 150 identifiants uniques, modeles ['bc_dag_all_10pct'], commits ['228a398']
