#!/usr/bin/env python3
"""
DOM-Studie: Sagt der Orderbuch-Zustand VOR t0 eine groessere Bewegung voraus -
und deren Richtung? (Horizonte 30/60/120 s)

Speicher-sicher: EIN Tag wird geladen, zur kleinen Sekunden-Tabelle verdichtet,
als Parquet gespeichert - danach werden die Rohdaten SOFORT verworfen. Es liegt
nie mehr als ein Tag Rohdaten im Speicher.

Nutzung:
  python run_dom_study.py --days 5                  # echte Daten ab START_DATE
  python run_dom_study.py --days 5 --start 2026-07-15
  python run_dom_study.py --days 3 --synth          # offline, ohne API-Key
  python run_dom_study.py --days 5 --force          # Tages-Cache neu berechnen
  python run_dom_study.py --analyze-only            # nur Analyse aus Cache
"""

from __future__ import annotations
import argparse
import gc
import sys

import numpy as np
import pandas as pd

from src import config as C
from src import chd_io as io
from src import dom
from src import synth as synthmod

SAMPLE_DIR = C.INTERIM_DIR / "dom_samples"
OUT = C.OUTPUT_DIR


def sample_path(date: str):
    return SAMPLE_DIR / f"{C.SYMBOL}_{date}_dom_sample.parquet"


def process_day(date: str, synth: bool = False, force: bool = False) -> pd.DataFrame:
    """Einen Tag verarbeiten -> kleine Sample-Tabelle. Rohdaten werden verworfen."""
    p = sample_path(date)
    if p.exists() and not force:
        print(f"  [{date}] Cache gefunden -> lade Sample (kein Download).")
        return io.load_parquet(p)

    print(f"\n  [{date}] lade Rohdaten ...")
    if synth:
        ob, tr = synthmod.make_synthetic(date=date, seed=abs(hash(date)) % 10000)
    else:
        ob = io.download_orderbook(date, date)
        tr = io.download_trades(date, date)
    ob_rows, tr_rows = len(ob), len(tr)
    ob_mb = ob.memory_usage(deep=True).sum() / 1e6
    print(f"  [{date}] Orderbuch: {ob_rows:,} Zeilen ({ob_mb:,.0f} MB) | Trades: {tr_rows:,} Zeilen")
    if ob.empty or tr.empty:
        print(f"  [{date}] WARNUNG: keine Daten -> Tag uebersprungen.")
        return pd.DataFrame()

    st = dom.build_second_table(ob, tr, date)
    sample = dom.build_sample(st, tr, date)

    # Anteil unklarer Mengenreduktionen ausweisen (Pull/Fill-Trennung).
    n_dec, n_amb = st["n_dec"].sum(), st["n_amb"].sum()
    amb_pct = (100.0 * n_amb / n_dec) if n_dec else 0.0

    # ---- Rohdaten SOFORT verwerfen ----
    del ob, tr, st
    gc.collect()

    io.save_parquet(sample, p)
    sm_mb = sample.memory_usage(deep=True).sum() / 1e6
    print(f"  [{date}] Sample: {len(sample):,} Sekunden-Zeitpunkte ({sm_mb:.1f} MB) "
          f"| unklare Reduktionen: {amb_pct:.1f}% | gespeichert: {p.name}")
    print(f"  [{date}] Rohdaten verworfen (Speicher freigegeben).")
    return sample


def dates_list(start: str, n: int):
    d0 = pd.Timestamp(start)
    return [(d0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:,.3f}")


def analyze(samples: dict[str, pd.DataFrame]):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)

    if not samples:
        print("Keine Daten zum Auswerten.")
        return

    pooled = pd.concat(samples.values(), ignore_index=True)

    print("\n" + "=" * 100)
    print("VORZEICHEN-KONVENTION")
    print("=" * 100)
    print("  imbalance = ((stack_bid + pull_ask) - (stack_ask + pull_bid)) / Summe aller vier")
    print("  POSITIV = Aufwaertsdruck : Bids werden aufgebaut und/oder Asks abgezogen.")
    print("  NEGATIV = Abwaertsdruck  : Asks werden aufgebaut und/oder Bids abgezogen.")
    print("  trade_imb = (buy_vol - sell_vol) / (buy_vol + sell_vol);  positiv = Kaeufer aggressiv.")
    print("  move_bps  = (mid[t0+h] / mid[t0] - 1) * 10000;  positiv = Preis gestiegen.")

    # ---------------- Kernfrage: Verteilung ----------------
    print("\n" + "=" * 100)
    print("KERNFRAGE 1: Verteilung der Vorwaertsbewegung |move| (bps) - ALLE Sekunden-Zeitpunkte")
    print("=" * 100)
    per_day = pd.concat([dom.move_distribution(s, d) for d, s in samples.items()],
                        ignore_index=True)
    print("\n  Pro Tag:")
    print(_fmt(per_day.sort_values(["horizon_s", "scope"])))
    pooled_dist = dom.move_distribution(pooled, "GEPOOLT")
    print("\n  Gepoolt:")
    print(_fmt(pooled_dist))

    control = dom.build_control(pooled)
    print("\n  Teilmengen (gepoolt):")
    subsets = [("aktive Phasen", pooled[pooled["active"]]),
               ("nahe VAH/VAL", pooled[pooled["near_level"]]),
               ("fern von VAH/VAL", pooled[~pooled["near_level"]]),
               ("KONTROLLE (zufaellig, aktiv)", control)]
    sub_dist = pd.concat([dom.move_distribution(s, nm) for nm, s in subsets if len(s)],
                         ignore_index=True)
    print(_fmt(sub_dist))

    # ---------------- Auswertung A: Groesse ----------------
    print("\n" + "=" * 100)
    print("AUSWERTUNG A: GROESSE - unterscheiden sich die Vorher-Merkmale bei grossem vs. kleinem Move?")
    print("=" * 100)
    size_pool = dom.size_analysis(pooled, "GEPOOLT")
    cols = ["pre_s", "horizon_s", "threshold", "n_big", "n_small", "belastbar",
            "imbalance__big", "imbalance__small", "trade_imb__big", "trade_imb__small",
            "vol__big", "vol__small", "spread__big", "spread__small",
            "depth__big", "depth__small"]
    print(_fmt(size_pool[cols]))
    print("\n  Stack/Pull je Seite (gepoolt):")
    cols2 = ["pre_s", "horizon_s", "threshold", "n_big",
             "stack_bid__big", "stack_bid__small", "stack_ask__big", "stack_ask__small",
             "pull_bid__big", "pull_bid__small", "pull_ask__big", "pull_ask__small"]
    print(_fmt(size_pool[cols2]))

    # ---------------- Auswertung B: Richtung ----------------
    print("\n" + "=" * 100)
    print("AUSWERTUNG B: RICHTUNG - nur unter den GROSSEN Moves: sagt die Vorher-Imbalance das Vorzeichen?")
    print("=" * 100)
    print("  base_up   = Anteil grosser Moves nach OBEN")
    print("  base_rate = max(base_up, 1-base_up) -> so gut waere schon 'immer dieselbe Richtung'.")
    print("  Eine Trefferquote zaehlt nur, wenn sie DEUTLICH ueber base_rate liegt (0.50 = Zufall).")
    dir_pool = dom.direction_analysis(pooled, "GEPOOLT")
    print(_fmt(dir_pool))

    dir_ctrl = dom.direction_analysis(control, "KONTROLLE")
    print("\n  Kontrollgruppe (zufaellige aktive Zeitpunkte, gleiche Messung):")
    print(_fmt(dir_ctrl[["scope", "pre_s", "horizon_s", "threshold", "n_big", "belastbar",
                         "base_rate", "hit_imbalance", "hit_trade_imb"]]))

    print("\n  Richtung pro Tag (Stabilitaet ueber Tage):")
    dir_days = pd.concat([dom.direction_analysis(s, d) for d, s in samples.items()],
                         ignore_index=True)
    stab = dir_days[(dir_days.pre_s == C.PRE_WINDOWS_S[1]) &
                    (dir_days.threshold == "taker_12bps")]
    print(_fmt(stab[["scope", "pre_s", "horizon_s", "threshold", "n_big", "belastbar",
                     "base_rate", "hit_imbalance", "hit_trade_imb"]]))

    # ---------------- Varianten & Speichern ----------------
    nv = dom.n_tested_variants()
    print("\n" + "=" * 100)
    print("GETESTETE VARIANTEN (Multiple Testing)")
    print("=" * 100)
    for k, v in nv.items():
        print(f"  {k}: {v}")
    total = (nv["groesse_zellen"] * nv["groesse_merkmale_pro_zelle"]
             + nv["richtung_zellen"] * nv["richtung_signale_pro_zelle"])
    print(f"  -> insgesamt rund {total} Einzelvergleiche (gepoolt), zusaetzlich je Tag.")
    print(f"  Bei so vielen Vergleichen sind einzelne 'Treffer' ohne Wiederholung "
          f"an neuen Tagen NICHT belastbar.")

    per_day.to_csv(OUT / "dom_move_distribution_per_day.csv", index=False)
    pooled_dist.to_csv(OUT / "dom_move_distribution_pooled.csv", index=False)
    sub_dist.to_csv(OUT / "dom_move_distribution_subsets.csv", index=False)
    size_pool.to_csv(OUT / "dom_size_analysis_pooled.csv", index=False)
    dir_pool.to_csv(OUT / "dom_direction_analysis_pooled.csv", index=False)
    dir_days.to_csv(OUT / "dom_direction_analysis_per_day.csv", index=False)
    dir_ctrl.to_csv(OUT / "dom_direction_analysis_control.csv", index=False)
    print(f"\n  CSV-Ergebnisse gespeichert in: {OUT}")

    # ---------------- Fazit zur Kernfrage ----------------
    print("\n" + "=" * 100)
    print("FAZIT ZUR KERNFRAGE (Kostenschwellen)")
    print("=" * 100)
    for _, r in pooled_dist.iterrows():
        print(f"  Horizont {int(r['horizon_s']):>3}s: Median |move| = {r['median']:.2f} bps | "
              f"ueber 6 bps (Maker): {100*r['frac_gt_6bps']:.1f}% | "
              f"ueber 12 bps (Taker): {100*r['frac_gt_12bps']:.1f}% der Zeitpunkte")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=C.STUDY_DAYS)
    ap.add_argument("--start", type=str, default=C.START_DATE)
    ap.add_argument("--synth", action="store_true", help="synthetische Daten (offline)")
    ap.add_argument("--force", action="store_true", help="Tages-Cache neu berechnen")
    ap.add_argument("--analyze-only", action="store_true", help="nur Analyse aus Cache")
    a = ap.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    dates = dates_list(a.start, a.days)

    print("=" * 100)
    print(f"DOM-STUDIE  |  {C.SYMBOL}  |  {a.days} Tage ab {a.start}")
    print(f"Horizonte: {C.HORIZONS_S} s | Vor-Fenster: {C.PRE_WINDOWS_S} s | "
          f"Schwellen: {C.THRESH_TAKER_BPS} bps (Taker) / {C.THRESH_MAKER_BPS} bps (Maker)")
    print("Verarbeitung: Tag fuer Tag, Rohdaten werden nach jedem Tag verworfen.")
    print("=" * 100)

    samples = {}
    for d in dates:
        if a.analyze_only:
            p = sample_path(d)
            if p.exists():
                samples[d] = io.load_parquet(p)
            continue
        try:
            s = process_day(d, synth=a.synth, force=a.force)
            if len(s):
                samples[d] = s
        except Exception as e:
            print(f"  [{d}] FEHLER: {type(e).__name__}: {e}")
        gc.collect()

    print(f"\nVerarbeitete Tage: {len(samples)} von {len(dates)}")
    analyze(samples)


if __name__ == "__main__":
    sys.exit(main())
