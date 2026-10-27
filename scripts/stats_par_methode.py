#!/usr/bin/env python3
"""
Statistiques SEPAREES pour BC seul et BC+HG-DAgger seul, a partir du CSV de la
campagne (results/experiments/trials_g1_final.csv).

Usage :
    python3 scripts/stats_par_methode.py results/experiments/trials_g1_40.csv
    python3 scripts/stats_par_methode.py results/experiments/trials_g1_40.csv --max-seed 20
        (--max-seed N : garde seulement les essais trial_seed < N, soit N essais par methode ;
         les poses de depart sont les memes pour les deux methodes)

Sortie : tableaux dans le terminal + results/experiments/stats_par_methode.csv
Aucune dependance externe (bibliotheque standard uniquement).
"""
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID = ("success", "collision", "timeout")


def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson a 95 % pour une proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fisher_two_sided(a, b, c, d):
    """Test exact de Fisher bilateral sur le tableau [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    p_obs = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:.2f} ± {sd:.2f}"


def summarize(rows):
    n = len(rows)
    out = Counter(r["outcome"] for r in rows)
    k = out["success"]
    lo, hi = wilson(k, n)
    succ = [r for r in rows if r["outcome"] == "success"]
    return {
        "n": n, "success": k, "collision": out["collision"], "timeout": out["timeout"],
        "rate": 100 * k / n if n else float("nan"),
        "ci_lo": 100 * lo, "ci_hi": 100 * hi,
        "nav_time_succ": mean_sd([fnum(r, "nav_time_s") for r in succ]),
        "path_succ": mean_sd([fnum(r, "path_length_m") for r in succ]),
        "path_all": mean_sd([fnum(r, "path_length_m") for r in rows]),
        "final_dist": mean_sd([fnum(r, "final_dist_m") for r in rows]),
        "safety": mean_sd([fnum(r, "safety_triggers") for r in rows]),
        "bumps": mean_sd([fnum(r, "n_bump_events") for r in rows]),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Statistiques par methode")
    ap.add_argument("csv")
    ap.add_argument("--max-seed", type=int, default=None,
                    help="garde seulement les essais dont trial_seed < N")
    args = ap.parse_args()
    path = Path(args.csv)
    all_rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    print(f"Fichier : {path}  ({len(all_rows)} lignes)")

    # --- controles ---
    uids = Counter(r["trial_uid"] for r in all_rows)
    dups = [u for u, c in uids.items() if c > 1]
    if dups:
        print(f"ATTENTION : {len(dups)} trial_uid en double (premiere occurrence gardee) : {dups[:3]}")
    seen, rows = set(), []
    for r in all_rows:
        if r["trial_uid"] in seen:
            continue
        seen.add(r["trial_uid"])
        rows.append(r)
    invalid = [r for r in rows if r["outcome"] not in VALID]
    if invalid:
        print(f"{len(invalid)} ligne(s) hors success/collision/timeout, ignoree(s) : "
              f"{Counter(r['outcome'] for r in invalid)}")
    rows = [r for r in rows if r["outcome"] in VALID]
    if args.max_seed is not None:
        rows = [r for r in rows if int(r["trial_seed"]) < args.max_seed]
        print(f"Filtre --max-seed {args.max_seed} : {len(rows)} essais gardes")

    methods = sorted({r["method"] for r in rows})
    by_method = {m: [r for r in rows if r["method"] == m] for m in methods}

    # --- tableau principal ---
    print("\n=== Resultats par methode ===")
    head = f"{'methode':16s} {'n':>3s} {'succes':>6s} {'coll.':>5s} {'timeout':>7s} {'taux':>7s}  {'IC 95 %':>15s}"
    print(head)
    print("-" * len(head))
    summ = {}
    for m in methods:
        s = summarize(by_method[m])
        summ[m] = s
        print(f"{m:16s} {s['n']:3d} {s['success']:6d} {s['collision']:5d} {s['timeout']:7d} "
              f"{s['rate']:6.1f}%  [{s['ci_lo']:4.0f}% - {s['ci_hi']:4.0f}%]")

    print("\n=== Details par methode (moyenne ± ecart-type) ===")
    for m in methods:
        s = summ[m]
        print(f"\n{m}")
        print(f"  temps de navigation (succes seulement) : {s['nav_time_succ']} s")
        print(f"  trajet, succes seulement               : {s['path_succ']} m")
        print(f"  trajet, tous les essais                : {s['path_all']} m")
        print(f"  distance finale au but                 : {s['final_dist']} m")
        print(f"  declenchements du filtre de securite   : {s['safety']}")
        print(f"  contacts (pare-chocs) par essai        : {s['bumps']}")

    # --- par seed d'entrainement ---
    print("\n=== Par methode et seed d'entrainement ===")
    for m in methods:
        print(f"{m}")
        seeds = defaultdict(list)
        for r in by_method[m]:
            seeds[r["train_seed"]].append(r)
        for sd in sorted(seeds):
            rr = seeds[sd]
            k = sum(r["outcome"] == "success" for r in rr)
            print(f"  seed {sd}: {k}/{len(rr)} succes ({100 * k / len(rr):.0f} %)")

    # --- comparaison des deux methodes ---
    if len(methods) == 2:
        a, b = methods
        sa, sb = summ[a], summ[b]
        p = fisher_two_sided(sa["success"], sa["n"] - sa["success"],
                             sb["success"], sb["n"] - sb["success"])
        print(f"\n=== Comparaison {a} vs {b} ===")
        print(f"  difference de taux de reussite : {sb['rate'] - sa['rate']:+.1f} points ({b} - {a})")
        print(f"  test exact de Fisher bilateral : p = {p:.3f}"
              f"  ({'difference significative a 5 %' if p < 0.05 else 'difference NON significative a 5 %'})")

        # essais apparies : meme trial_seed = meme pose de depart
        pairs = defaultdict(dict)
        for r in rows:
            pairs[r["trial_seed"]][r["method"]] = r["outcome"] == "success"
        both = [v for v in pairs.values() if a in v and b in v]
        if both:
            only_b = sum(1 for v in both if v[b] and not v[a])
            only_a = sum(1 for v in both if v[a] and not v[b])
            print(f"  essais apparies (meme pose de depart) : {len(both)}")
            print(f"    reussi seulement par {b} : {only_b}")
            print(f"    reussi seulement par {a} : {only_a}")
            print(f"    reussi par les deux : {sum(1 for v in both if v[a] and v[b])} | "
                  f"echec des deux : {sum(1 for v in both if not v[a] and not v[b])}")

    # --- export ---
    out = path.with_name("stats_par_methode" + (f"_seed{args.max_seed}" if args.max_seed else "") + ".csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["methode", "n", "succes", "collisions", "timeouts", "taux_%", "ic95_bas_%", "ic95_haut_%"])
        for m in methods:
            s = summ[m]
            w.writerow([m, s["n"], s["success"], s["collision"], s["timeout"],
                        round(s["rate"], 1), round(s["ci_lo"], 1), round(s["ci_hi"], 1)])
    print(f"\nExport : {out}")


if __name__ == "__main__":
    main()