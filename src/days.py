"""
Tagesweise Datenbeschaffung mit persistenter Sekundentabelle.

Warum: Ein Tag Orderbuch sind ~3 GB. Die daraus verdichtete Sekundentabelle ist
nur ~8 MB und enthaelt alles, was die Auswertungen brauchen. Wird sie
gespeichert, kommt JEDE weitere Studie ohne erneuten Download aus.

ensure_seconds(date):
  - Sekundentabelle im Cache?  -> laden (kein Download)
  - sonst: EINEN Tag laden -> verdichten -> speichern -> Rohdaten sofort verwerfen
Es liegt nie mehr als ein Tag Rohdaten im Speicher.
"""

from __future__ import annotations
import gc

import pandas as pd

from . import config as C
from . import chd_io as io
from . import dom
from . import synth as synthmod

SECONDS_DIR = C.INTERIM_DIR / "dom_seconds"


def seconds_path(date: str):
    return SECONDS_DIR / f"{C.SYMBOL}_{date}_seconds.parquet"


def load_raw(date: str, synth: bool = False):
    """Rohdaten eines Tages holen (echt oder synthetisch)."""
    if synth:
        return synthmod.make_synthetic(date=date, seed=abs(hash(date)) % 10000)
    return io.download_orderbook(date, date), io.download_trades(date, date)


def ensure_seconds(date: str, synth: bool = False, force: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """Sekundentabelle des Tages liefern; bei Bedarf einmalig aus Rohdaten bauen."""
    SECONDS_DIR.mkdir(parents=True, exist_ok=True)
    p = seconds_path(date)
    if p.exists() and not force:
        if verbose:
            print(f"  [{date}] Sekundentabelle aus Cache (kein Download).")
        return io.load_parquet(p)

    if verbose:
        print(f"  [{date}] lade Rohdaten ...")
    ob, tr = load_raw(date, synth=synth)
    if ob.empty or tr.empty:
        if verbose:
            print(f"  [{date}] WARNUNG: keine Daten.")
        return pd.DataFrame()
    if verbose:
        mb = ob.memory_usage(deep=True).sum() / 1e6
        print(f"  [{date}] Orderbuch: {len(ob):,} Zeilen ({mb:,.0f} MB) | "
              f"Trades: {len(tr):,} Zeilen")

    st = dom.build_second_table(ob, tr, date)
    n_dec, n_amb = st["n_dec"].sum(), st["n_amb"].sum()
    amb_pct = (100.0 * n_amb / n_dec) if n_dec else 0.0

    del ob, tr
    gc.collect()

    io.save_parquet(st, p)
    if verbose:
        print(f"  [{date}] Sekundentabelle: {len(st):,} Sekunden "
              f"({st.memory_usage(deep=True).sum()/1e6:.1f} MB) | "
              f"unklare Reduktionen: {amb_pct:.1f}% -> {p.name}")
        print(f"  [{date}] Rohdaten verworfen (Speicher freigegeben).")
    return st


def dates_list(start: str, n: int):
    d0 = pd.Timestamp(start)
    return [(d0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]
