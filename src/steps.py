"""
Orchestrierung: eine Funktion je Schritt. Jede Funktion
  - liest ihren Input (aus Parquet-Cache),
  - rechnet,
  - speichert ihr Ergebnis als Parquet (damit nichts doppelt geladen wird),
  - druckt eine Diagnose (Zeilen, Speicher, Zeitraum).

`synth=True` nutzt kuenstliche Testdaten (offline, ohne API-Key).
`force=True` ignoriert den Cache und rechnet neu.
"""

from __future__ import annotations
import pandas as pd

from . import config as C
from . import chd_io as io
from . import pipeline as P
from . import book
from . import synth as synthmod

T = C.tag()
P_OB   = C.RAW_DIR / f"{T}_orderbook.parquet"
P_TR   = C.RAW_DIR / f"{T}_trades.parquet"
P_VA   = C.INTERIM_DIR / f"{T}_value_area.parquet"
P_REF  = C.INTERIM_DIR / f"{T}_refmap.parquet"
P_EV   = C.INTERIM_DIR / f"{T}_events.parquet"
P_TOB  = C.INTERIM_DIR / f"{T}_tob.parquet"
P_EVM  = C.INTERIM_DIR / f"{T}_events_measured.parquet"
P_CTM  = C.INTERIM_DIR / f"{T}_controls_measured.parquet"
P_REP  = C.OUTPUT_DIR / f"{T}_report.parquet"
P_REPC = C.OUTPUT_DIR / f"{T}_report.csv"


def _hdr(n, title):
    print("\n" + "=" * 72)
    print(f"SCHRITT {n}: {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
def step01_load(synth: bool = False, force: bool = False):
    _hdr(1, "Rohdaten laden (Orderbuch + Trades)")
    if P_OB.exists() and P_TR.exists() and not force:
        print("  Cache gefunden -> lade Parquet (kein erneuter Download).")
        ob, tr = io.load_parquet(P_OB), io.load_parquet(P_TR)
    else:
        if synth:
            print("  Modus: SYNTHETISCH (offline, ohne API-Key).")
            ob, tr = synthmod.make_synthetic()
        else:
            print("  Modus: ECHTER DOWNLOAD via cryptohftdata.")
            if not io.get_api_key():
                print("  WARNUNG: kein CRYPTOHFTDATA_API_KEY in .env -> Free-Tier (rate-limited).")
            ob, tr = io.download_orderbook(), io.download_trades()
        io.save_parquet(ob, P_OB)
        io.save_parquet(tr, P_TR)
    io.describe_df(ob, "Orderbuch (roh, kanonisch)")
    if "event_type" in ob.columns and len(ob):
        print(f"  event_type-Verteilung: {ob['event_type'].value_counts().to_dict()}")
    io.describe_df(tr, "Trades (roh, kanonisch)")
    return ob, tr


# ---------------------------------------------------------------------------
def step02_value_area(force: bool = False):
    _hdr(2, "Value Area je Session + Referenz-Zuordnung (kein Look-ahead)")
    tr = io.load_parquet(P_TR)
    va = P.compute_sessions_va(tr)
    ref = P.build_reference_map(va)
    io.save_parquet(va, P_VA)
    io.save_parquet(ref, P_REF)
    print("\n  Value Area je Session:")
    print(va.to_string(index=False))
    print("\n  Referenz-Zuordnung (Ziel-Session <- Vor-Session):")
    print(ref.to_string(index=False) if len(ref) else "  (leer)")
    io.describe_df(va, "value_area", time_col="start")
    return va, ref


# ---------------------------------------------------------------------------
def step03_events(force: bool = False):
    _hdr(3, "Events: Preis erreicht Referenzlevel (VAH/VAL der Vor-Session)")
    tr = io.load_parquet(P_TR)
    ref = io.load_parquet(P_REF)
    ev = P.detect_events(tr, ref)
    io.save_parquet(ev, P_EV)
    io.describe_df(ev, "events", time_col="t0")
    if len(ev):
        print(f"  Events nach Level-Typ: {ev['level_type'].value_counts().to_dict()}")
        print(f"  Events nach Session  : {ev['session'].value_counts().to_dict()}")
    return ev


# ---------------------------------------------------------------------------
def step04_top_of_book(force: bool = False):
    _hdr(4, "Top-of-Book rekonstruieren -> Spread (fuer Kosten)")
    ob = io.load_parquet(P_OB)
    tob = book.reconstruct_top_of_book(ob)
    io.save_parquet(tob, P_TOB)
    io.describe_df(tob, "top_of_book (1 Zeile/Sek.)")
    sp = P.mean_spread_bps(tob)
    print(f"  Durchschnittlicher Spread: {sp:.2f} bps")
    return tob


# ---------------------------------------------------------------------------
def step05_measure(force: bool = False):
    _hdr(5, "Messung: Pull/Stack-Imbalance (vorher) + Move (nachher), Events & Kontrolle")
    ob = io.load_parquet(P_OB)
    tr = io.load_parquet(P_TR)
    ev = io.load_parquet(P_EV)
    ref = io.load_parquet(P_REF)

    obd = book.level_deltas(ob)
    pp = P.PricePath(tr)
    meas = P.Measurer(obd, tr, pp)

    ev_meas = P.measure_events(ev, meas)
    ct_meas = P.build_controls(tr, ref, meas)

    io.save_parquet(ev_meas if len(ev_meas) else pd.DataFrame({"_empty": []}), P_EVM)
    io.save_parquet(ct_meas if len(ct_meas) else pd.DataFrame({"_empty": []}), P_CTM)

    io.describe_df(ev_meas, "events_measured", time_col="t0")
    io.describe_df(ct_meas, "controls_measured", time_col="t0")

    # Anteil unklarer (ambiguous) Mengenreduktionen ausweisen (Anforderung 1).
    for nm, df in [("Events", ev_meas), ("Kontrolle", ct_meas)]:
        if len(df):
            n_dec = df["n_dec"].sum(); n_amb = df["n_amb"].sum()
            pct = (100.0 * n_amb / n_dec) if n_dec else 0.0
            print(f"  {nm}: Mengenreduktionen={int(n_dec)}, davon unklar (ambiguous)={int(n_amb)} = {pct:.1f}%")
    return ev_meas, ct_meas


# ---------------------------------------------------------------------------
def step06_report(force: bool = False):
    _hdr(6, "Report: Fallzahlen + Imbalance Event vs. Kontrolle + Kostenvergleich")
    ev_meas = io.load_parquet(P_EVM)
    ct_meas = io.load_parquet(P_CTM)
    tob = io.load_parquet(P_TOB)
    if "_empty" in ev_meas.columns:
        ev_meas = pd.DataFrame()
    if "_empty" in ct_meas.columns:
        ct_meas = pd.DataFrame()

    rep = P.build_report(ev_meas, ct_meas, tob)
    io.save_parquet(rep, P_REP)
    rep.to_csv(P_REPC, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + rep.to_string(index=False))
    print("\n  Hinweise:")
    print("  - 'belastbar' = >=30 Faelle in BEIDEN Gruppen (sonst nicht belastbar).")
    print(f"  - 'handelbar' = mittlerer |Move| > Kosten (Kosten = {C.ROUND_TRIP_FEE_BPS:.0f} bps "
          f"Taker rund + Spread).")
    if len(rep):
        print(f"  - Getestete Varianten (Multiple Testing): {int(rep['n_tests_total'].iloc[0])} "
              f"(= {len(C.PRE_WINDOWS_S)} Vor-Fenster x {len(C.POST_HORIZONS_S)} Nach-Fenster).")
    print(f"\n  Report gespeichert: {P_REP}")
    print(f"  Report als CSV    : {P_REPC}")
    return rep


# ---------------------------------------------------------------------------
def run_all(synth: bool = False, force: bool = False):
    step01_load(synth=synth, force=force)
    step02_value_area(force=force)
    step03_events(force=force)
    step04_top_of_book(force=force)
    step05_measure(force=force)
    step06_report(force=force)
    print("\nFERTIG. Alle Zwischenstaende liegen als Parquet in data/interim/ bzw. data/output/.")
