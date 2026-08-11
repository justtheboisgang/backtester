#!/usr/bin/env python3
"""
Ehrliche Nachrechnung zur DOM-Studie (laeuft auf den gecachten Tages-Samples).

Behebt zwei Dinge, die im Hauptreport zu optimistisch dargestellt sind:

1) EFFEKTIVE FALLZAHL. Die Sekunden-Zeitpunkte ueberlappen: bei 120 s Horizont
   teilen sich 120 aufeinanderfolgende Zeitpunkte fast denselben Move. "n=44.325"
   sind daher KEINE 44.325 unabhaengigen Beobachtungen. Hier wird auf
   nicht-ueberlappende Bloecke ausgeduennt (Schrittweite = Horizont) und mit
   ehrlichem Konfidenzintervall gerechnet.

2) ERWARTUNGSWERT NACH KOSTEN. Eine Trefferquote allein sagt nichts ueber Geld.
   Gerechnet werden zwei Varianten:
     (a) REALISTISCH: an JEDEM Zeitpunkt auf sign(Signal) setzen. Ob der Move
         gross wird, weiss man vorher nicht.
     (b) ORAKEL (nicht handelbar): nur unter den bereits grossen Moves setzen.
         Das ist die theoretische Obergrenze, nicht erreichbar.

Nutzung:  python analyze_edge.py [--days 5] [--start 2026-07-15]
"""

from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from src import config as C
from src import chd_io as io

SAMPLE_DIR = C.INTERIM_DIR / "dom_samples"
SIGNALS = ["imbalance", "trade_imb"]
COSTS = {"taker_12bps": C.THRESH_TAKER_BPS, "maker_6bps": C.THRESH_MAKER_BPS}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson-Konfidenzintervall fuer eine Trefferquote (robust bei kleinem n)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - hw, c + hw)


def load_samples(dates):
    out = {}
    for d in dates:
        p = SAMPLE_DIR / f"{C.SYMBOL}_{d}_dom_sample.parquet"
        if p.exists():
            out[d] = io.load_parquet(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=C.STUDY_DAYS)
    ap.add_argument("--start", type=str, default=C.START_DATE)
    a = ap.parse_args()

    d0 = pd.Timestamp(a.start)
    dates = [(d0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(a.days)]
    samples = load_samples(dates)
    if not samples:
        print("Keine gecachten Samples gefunden. Erst run_dom_study.py laufen lassen.")
        return
    pooled = pd.concat(samples.values(), ignore_index=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    print("=" * 104)
    print("EFFEKTIVE FALLZAHL: ueberlappende Fenster ausgeduennt (Schrittweite = Horizont)")
    print("=" * 104)
    rows = []
    for h in C.HORIZONS_S:
        indep = pooled[pooled["t0_sec"] % h == 0]
        for thr_name, thr in COSTS.items():
            n_big = int((indep[f"absmove_{h}"] > thr).sum())
            rows.append({"horizon_s": h, "threshold": thr_name,
                         "n_roh": int((pooled[f"absmove_{h}"] > thr).sum()),
                         "n_unabhaengig": n_big,
                         "belastbar": n_big >= 30})
    eff = pd.DataFrame(rows)
    print(eff.to_string(index=False))

    print("\n" + "=" * 104)
    print("RICHTUNG auf unabhaengigen Bloecken (mit 95%-Konfidenzintervall)")
    print("=" * 104)
    print("  base_rate = so gut waere schon 'immer dieselbe Richtung'.")
    print("  Ein Signal zaehlt nur, wenn die UNTERGRENZE des CI ueber base_rate liegt.\n")
    rows = []
    for w in C.PRE_WINDOWS_S:
        for h in C.HORIZONS_S:
            indep = pooled[pooled["t0_sec"] % h == 0]
            for thr_name, thr in COSTS.items():
                big = indep[indep[f"absmove_{h}"] > thr]
                n_big = len(big)
                if n_big == 0:
                    continue
                sgn = np.sign(big[f"move_{h}"].to_numpy())
                base_up = float((sgn > 0).mean())
                rec = {"pre_s": w, "horizon_s": h, "threshold": thr_name, "n": n_big,
                       "belastbar": n_big >= 30,
                       "base_rate": round(max(base_up, 1 - base_up), 3)}
                for sig in SIGNALS:
                    x = big[f"{sig}_{w}"].to_numpy()
                    use = np.abs(x) > 1e-12
                    n_use = int(use.sum())
                    k = int((np.sign(x[use]) == sgn[use]).sum())
                    lo, hi = wilson(k, n_use)
                    rec[f"{sig}_hit"] = round(k / n_use, 3) if n_use else np.nan
                    rec[f"{sig}_ci"] = f"[{lo:.3f},{hi:.3f}]" if n_use else "-"
                rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 104)
    print("ERWARTUNGSWERT NACH KOSTEN (bps je Trade, unabhaengige Bloecke)")
    print("=" * 104)
    print("  REALISTISCH: an jedem Zeitpunkt auf sign(Signal) setzen, Kosten immer zahlen.")
    print("  ORAKEL     : nur unter bereits grossen Moves - NICHT handelbar, nur Obergrenze.\n")
    rows = []
    for w in C.PRE_WINDOWS_S:
        for h in C.HORIZONS_S:
            indep = pooled[pooled["t0_sec"] % h == 0]
            for sig in SIGNALS:
                x = indep[f"{sig}_{w}"].to_numpy()
                mv = indep[f"move_{h}"].to_numpy()
                use = np.abs(x) > 1e-12
                gross = np.sign(x[use]) * mv[use]
                n = len(gross)
                if n == 0:
                    continue
                se = gross.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                for cost_name, cost in COSTS.items():
                    net = gross.mean() - cost
                    rec = {"pre_s": w, "horizon_s": h, "signal": sig, "kosten": cost_name,
                           "n": n, "brutto_bps": round(float(gross.mean()), 3),
                           "netto_bps": round(float(net), 3),
                           "brutto_se": round(float(se), 3),
                           "profitabel": bool(net > 0)}
                    # Orakel-Obergrenze
                    big = indep[indep[f"absmove_{h}"] > COSTS[cost_name]]
                    xb = big[f"{sig}_{w}"].to_numpy()
                    mb = big[f"move_{h}"].to_numpy()
                    ub = np.abs(xb) > 1e-12
                    if ub.sum():
                        og = np.sign(xb[ub]) * mb[ub]
                        rec["orakel_netto_bps"] = round(float(og.mean() - cost), 3)
                        rec["orakel_n"] = int(ub.sum())
                    rows.append(rec)
    ev = pd.DataFrame(rows)
    print(ev.to_string(index=False))

    ev.to_csv(C.OUTPUT_DIR / "dom_edge_after_costs.csv", index=False)
    print(f"\n  gespeichert: {C.OUTPUT_DIR / 'dom_edge_after_costs.csv'}")

    print("\n" + "=" * 104)
    print("FAZIT")
    print("=" * 104)
    best = ev.sort_values("netto_bps", ascending=False).iloc[0]
    print(f"  Beste realistische Variante von {len(ev)} getesteten:")
    print(f"    pre={int(best['pre_s'])}s horizon={int(best['horizon_s'])}s "
          f"signal={best['signal']} kosten={best['kosten']}")
    print(f"    brutto {best['brutto_bps']:.3f} bps (SE {best['brutto_se']:.3f}) "
          f"-> netto {best['netto_bps']:.3f} bps je Trade")
    print(f"  Da {len(ev)} Varianten geprueft wurden, ist die BESTE davon systematisch")
    print(f"  zu optimistisch - sie ist als Schaetzer nicht verwendbar.")


if __name__ == "__main__":
    main()
