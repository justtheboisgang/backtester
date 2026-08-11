#!/usr/bin/env python3
"""
Volatilitaets-Studie: Sagt der Orderbuch-Zustand die realisierte Volatilitaet der
naechsten 30/60/120 s vorher - ueber vergangene Volatilitaet hinaus?

Speicher-sicher: Tag fuer Tag. Ist die Sekundentabelle eines Tages schon
gespeichert (aus einem frueheren Lauf), wird NICHT erneut heruntergeladen.

Nutzung:
  python run_vola_study.py --days 5
  python run_vola_study.py --days 3 --synth      # offline, ohne API-Key
  python run_vola_study.py --days 5 --force      # Sekundentabellen neu bauen
"""

from __future__ import annotations
import argparse
import gc

import pandas as pd

from src import config as C
from src import days as D
from src import vola

OUT = C.OUTPUT_DIR


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:,.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=C.STUDY_DAYS)
    ap.add_argument("--start", type=str, default=C.START_DATE)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    dates = D.dates_list(a.start, a.days)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    print("=" * 104)
    print(f"VOLATILITAETS-STUDIE  |  {C.SYMBOL}  |  {a.days} Tage ab {a.start}")
    print(f"Horizonte: {C.HORIZONS_S} s | Vor-Fenster: {C.PRE_WINDOWS_S} s")
    print("Verarbeitung: Tag fuer Tag; Sekundentabellen werden wiederverwendet.")
    print("=" * 104)

    samples = {}
    for d in dates:
        try:
            st = D.ensure_seconds(d, synth=a.synth, force=a.force)
            if st.empty:
                continue
            s = vola.build_vola_sample(st, d)
            del st
            gc.collect()
            if len(s):
                samples[d] = s
                print(f"  [{d}] Vola-Sample: {len(s):,} Zeitpunkte "
                      f"({s.memory_usage(deep=True).sum()/1e6:.1f} MB)")
        except Exception as e:
            print(f"  [{d}] FEHLER: {type(e).__name__}: {e}")
        gc.collect()

    if not samples:
        print("\nKeine Daten. Abbruch.")
        return
    print(f"\nVerarbeitete Tage: {len(samples)} von {len(dates)}")

    # ---------------- Zielgroessen ----------------
    print("\n" + "=" * 104)
    print("ZIELGROESSEN: realisierte Volatilitaet NACH t0 (unabhaengige Bloecke)")
    print("=" * 104)
    print("  rv    = Standardabweichung der Sekunden-Returns ueber den Horizont (bps)")
    print("  range = High-Low-Spanne des Mid ueber den Horizont, relativ zu mid[t0] (bps)\n")
    rows = []
    for h in C.HORIZONS_S:
        for d, s in samples.items():
            b = s[s["t0_sec"] % h == 0]
            rows.append({"scope": d, "horizon_s": h, "n_eff": len(b),
                         "rv_median": b[f"rv_{h}"].median(), "rv_mittel": b[f"rv_{h}"].mean(),
                         "range_median": b[f"range_{h}"].median(),
                         "range_mittel": b[f"range_{h}"].mean()})
        allb = pd.concat([s[s["t0_sec"] % h == 0] for s in samples.values()], ignore_index=True)
        rows.append({"scope": "GEPOOLT", "horizon_s": h, "n_eff": len(allb),
                     "rv_median": allb[f"rv_{h}"].median(), "rv_mittel": allb[f"rv_{h}"].mean(),
                     "range_median": allb[f"range_{h}"].median(),
                     "range_mittel": allb[f"range_{h}"].mean()})
    tgt = pd.DataFrame(rows)
    print(_fmt(tgt.sort_values(["horizon_s", "scope"])))

    # ---------------- Effektive Fallzahl ----------------
    print("\n" + "=" * 104)
    print("EFFEKTIVE FALLZAHL (ueberlappende Fenster ausgeduennt)")
    print("=" * 104)
    rows = []
    for h in C.HORIZONS_S:
        roh = sum(len(s) for s in samples.values())
        eff = sum(len(s[s["t0_sec"] % h == 0]) for s in samples.values())
        rows.append({"horizon_s": h, "n_roh_sekunden": roh, "n_unabhaengige_bloecke": eff,
                     "faktor": round(roh / eff, 1) if eff else None})
    print(_fmt(pd.DataFrame(rows)))
    print("  Nur die rechte Spalte zaehlt statistisch.")

    # ---------------- Die Treppe ----------------
    print("\n" + "=" * 104)
    print("DIE TREPPE (Leave-one-day-out, also OUT-OF-SAMPLE bewertet)")
    print("=" * 104)
    print("  S1 = nur vergangene Volatilitaet")
    print("  S2 = S1 + gehandeltes Volumen + Trade-Anzahl")
    print("  S3 = S2 + Orderbuch (Netto-Abfluss, Tiefenveraenderung, Umschichtung, Tiefe, Spread)")
    print("  d_r2 / d_mae = Verbesserung gegenueber der VORIGEN Stufe.")
    print("  WICHTIG: in-sample steigt R2 immer, wenn man Merkmale hinzufuegt. Deshalb wird")
    print("  auf einem ausgelassenen Tag getestet - nur das ist ein echter Beleg.\n")
    lad = vola.ladder(samples)
    cols = ["pre_s", "horizon_s", "ziel", "stufe", "n_eff", "belastbar",
            "r2_oos", "mae_bps", "d_r2_vs_vorstufe", "d_mae_vs_vorstufe"]
    print(_fmt(lad[cols]))

    print("\n  Zusammenfassung: bringt Stufe 3 gegenueber Stufe 2 etwas?")
    s3 = lad[lad.stufe == "S3_plus_Orderbuch"]
    n_better = int((s3["d_r2_vs_vorstufe"] > 0).sum())
    print(f"    In {n_better} von {len(s3)} Zellen ist R2(S3) > R2(S2).")
    print(f"    Mittlere Verbesserung R2 durch das Orderbuch: "
          f"{s3['d_r2_vs_vorstufe'].mean():+.5f}")
    print(f"    Mittlere Veraenderung MAE (negativ = besser):  "
          f"{s3['d_mae_vs_vorstufe'].mean():+.5f} bps")
    s2 = lad[lad.stufe == "S2_plus_Volumen"]
    print(f"    Zum Vergleich, Volumen gegenueber nur Vola (S2 vs S1): R2 "
          f"{s2['d_r2_vs_vorstufe'].mean():+.5f}")

    # ---------------- Terzile ----------------
    print("\n" + "=" * 104)
    print("TERZILE NACH NETTO-ABFLUSS  (Grenzen je Tag bestimmt)")
    print("=" * 104)
    print("  netout = (pull_bid + pull_ask) - (stack_bid + stack_ask)")
    print("  positiv = mehr Liquiditaet abgezogen als aufgebaut (abfliessend)\n")
    ter = vola.terciles(samples)
    print(_fmt(ter))

    ter_day = vola.terciles(samples, per_day=True)

    print("\n  Traegt netout eigenstaendige Information? (Diagnose)")
    print("  netout zaehlt bei Abgaengen NUR Stornos - Ausfuehrungen sind herausklassifiziert.")
    print("  Ueber ein Fenster gilt: Tiefenaenderung = stack - pull - fills. Bleibt die Tiefe")
    print("  konstant, folgt netout ~ -Handelsvolumen. Dann misst netout im Kern nur Volumen.")
    diag = vola.netout_diagnosis(samples)
    print(_fmt(diag))
    worst = diag["corr_netout_volumen"].min()
    if worst < -0.5:
        print(f"  -> Korrelation bis {worst:.2f}: netout ist weitgehend ein Volumen-Abbild,")
        print("     kein eigenstaendiges Orderbuch-Signal.")

    # ---------------- Varianten ----------------
    print("\n" + "=" * 104)
    print("GETESTETE VARIANTEN")
    print("=" * 104)
    for k, v in vola.n_variants(len(samples)).items():
        print(f"  {k}: {v}")
    print("  Keine Parametersuche: Fenster, Horizonte und Merkmale standen vorher fest.")

    tgt.to_csv(OUT / "vola_targets.csv", index=False)
    lad.to_csv(OUT / "vola_ladder.csv", index=False)
    ter.to_csv(OUT / "vola_terciles_pooled.csv", index=False)
    ter_day.to_csv(OUT / "vola_terciles_per_day.csv", index=False)
    diag.to_csv(OUT / "vola_netout_diagnosis.csv", index=False)
    print(f"\n  CSV-Ergebnisse gespeichert in: {OUT}")


if __name__ == "__main__":
    main()
