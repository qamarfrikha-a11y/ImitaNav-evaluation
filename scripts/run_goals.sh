#!/bin/bash
# Lance la campagne d'evaluation pour un ou plusieurs goals, avec reprise automatique
# (run_campaign.sh saute les essais deja presents dans le fichier de resultats).
#
# Usage :
#   ./scripts/run_goals.sh G2 G3 G4 G5
#   N=30 TAG=dagall ./scripts/run_goals.sh G1 G2 G3 G4 G5
#   DRY_RUN=1 TAG=dagall ./scripts/run_goals.sh G1       # affiche sans rien lancer
#
# Variables :
#   N    nombre d'essais par methode du plan (defaut 30)
#   TAG  suffixe du plan et des resultats (defaut : aucun, resultats en *_v2.csv)
#
# Entrees  : results/experiments/plan_<goal>_<N>[_<TAG>].csv   (cree par make_plan_goal.py)
# Sorties  : results/experiments/trials_<goal>_<TAG ou v2>.csv

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
N="${N:-30}"
TAG="${TAG:-}"

[[ $# -ge 1 ]] || { echo "Usage: $0 G1 [G2 G3 G4 G5]"; exit 1; }

for goal in "$@"; do
    g="$(echo "$goal" | tr 'A-Z' 'a-z')"
    if [[ -n "$TAG" ]]; then
        plan="results/experiments/plan_${g}_${N}_${TAG}.csv"
        out="results/experiments/trials_${g}_${TAG}.csv"
    else
        plan="results/experiments/plan_${g}_${N}.csv"
        out="results/experiments/trials_${g}_v2.csv"
    fi
    if [[ ! -f "$plan" ]]; then
        echo "Plan introuvable : $plan (lancer d'abord make_plan_goal.py)"
        continue
    fi
    echo ""
    echo "##### $goal : $plan -> $out"
    ./scripts/run_campaign.sh "$plan" "$out"
done
