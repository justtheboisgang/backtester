"""
Laden, Normalisieren, Diagnostizieren und Cachen der Daten.

Wichtig: Die ECHTEN Spaltennamen von cryptohftdata koennen je nach Exchange
leicht abweichen (das sagt das Paket selbst). Deshalb ist der Loader
"schema-tolerant": er erkennt die vorhandenen Spalten zur Laufzeit und
uebersetzt sie auf ein festes, kanonisches Schema:

  Orderbuch (ob):   event_time | event_type | side('bid'/'ask') | price | quantity
                    quantity = absolute Restmenge auf dem Level; 0 = Level geloescht.
  Trades   (tr):    event_time | side('buy'/'sell') | price | quantity

Beim ersten echten Download werden die ROH-Spalten ausgedruckt, damit du (und
ich beim naechsten Lauf) sehen, was der Anbieter tatsaechlich liefert.
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cryptohftdata as chd

from . import config as C


# ===========================================================================
#  .env / API-Key
# ===========================================================================
def load_dotenv(path: str | Path = None) -> None:
    """Winziger .env-Parser (keine Extra-Abhaengigkeit). Setzt os.environ."""
    path = Path(path) if path else (C.ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def get_api_key() -> str | None:
    load_dotenv()
    return os.environ.get("CRYPTOHFTDATA_API_KEY")


# ===========================================================================
#  Diagnostik  (Anforderung: nach jedem Schritt Zeilenzahl, Speicher, Zeitraum)
# ===========================================================================
def describe_df(df: pd.DataFrame, name: str, time_col: str = "event_time") -> dict:
    """Druckt eine kompakte Diagnose und gibt sie als dict zurueck."""
    n = len(df)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"\n--- Diagnose: {name} ---")
    print(f"  Zeilen      : {n:,}")
    print(f"  Speicher    : {mem_mb:,.1f} MB")
    info = {"name": name, "rows": n, "mem_mb": round(mem_mb, 1)}
    if time_col in df.columns and n:
        t0, t1 = df[time_col].min(), df[time_col].max()
        dur = t1 - t0
        print(f"  Zeitraum    : {t0}  ->  {t1}  (Dauer {dur})")
        info["t_start"], info["t_end"] = str(t0), str(t1)
    print(f"  Spalten ({len(df.columns)}): {list(df.columns)}")
    return info


# ===========================================================================
#  Spalten-Normalisierung
# ===========================================================================
def _pick(columns, candidates):
    """Finde die erste vorhandene Spalte aus einer Kandidatenliste (case-insensitiv)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _to_utc_datetime(s: pd.Series) -> pd.Series:
    """Zeitspalte robust nach UTC-datetime konvertieren (ms/us/ns-Epoch oder String)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, utc=True)
        return out
    if pd.api.types.is_numeric_dtype(s):
        # Epoch-Einheit anhand der Groessenordnung raten.
        v = float(pd.to_numeric(s.dropna().iloc[0])) if len(s.dropna()) else 0.0
        if v > 1e17:      unit = "ns"
        elif v > 1e14:    unit = "us"
        elif v > 1e11:    unit = "ms"
        else:             unit = "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True)


def _norm_side_ob(s: pd.Series) -> pd.Series:
    """Orderbuch-Seite -> 'bid'/'ask'."""
    m = {"bid": "bid", "b": "bid", "buy": "bid", "0": "bid",
         "ask": "ask", "a": "ask", "sell": "ask", "1": "ask"}
    return s.astype(str).str.strip().str.lower().map(m).fillna(s.astype(str).str.lower())


def _norm_side_trade(s: pd.Series) -> pd.Series:
    """Trade-Aggressor-Seite -> 'buy'/'sell'."""
    m = {"buy": "buy", "b": "buy", "bid": "buy",
         "sell": "sell", "s": "sell", "ask": "sell"}
    return s.astype(str).str.strip().str.lower().map(m).fillna(s.astype(str).str.lower())


def normalize_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    """Roh-Orderbuch -> kanonisches Schema. Druckt die Roh-Spalten zur Kontrolle."""
    print(f"  [normalize_orderbook] ROH-Spalten: {list(df.columns)}")
    if df.empty:
        return pd.DataFrame(columns=["event_time", "event_type", "side", "price", "quantity"])
    cols = df.columns
    c_time = _pick(cols, ["event_time", "timestamp", "time", "ts", "received_time", "local_time"])
    c_side = _pick(cols, ["side", "is_bid"])
    c_price = _pick(cols, ["price", "px"])
    c_qty = _pick(cols, ["quantity", "qty", "size", "amount", "new_quantity"])
    c_type = _pick(cols, ["event_type", "type", "update_type", "action"])
    missing = [n for n, c in [("time", c_time), ("side", c_side),
                              ("price", c_price), ("quantity", c_qty)] if c is None]
    if missing:
        raise ValueError(
            f"Orderbuch: konnte Spalte(n) {missing} nicht zuordnen. "
            f"Vorhandene Spalten: {list(cols)}. Bitte _pick-Kandidaten in chd_io.py ergaenzen."
        )
    out = pd.DataFrame({
        "event_time": _to_utc_datetime(df[c_time]),
        "side": _norm_side_ob(df[c_side]),
        "price": pd.to_numeric(df[c_price], errors="coerce").astype("float64"),
        "quantity": pd.to_numeric(df[c_qty], errors="coerce").astype("float64"),
    })
    out["event_type"] = (df[c_type].astype(str).str.lower() if c_type
                         else pd.Series("update", index=df.index))
    out = out[["event_time", "event_type", "side", "price", "quantity"]]
    out = out.dropna(subset=["event_time", "price", "quantity"])
    out = out.sort_values("event_time", kind="stable").reset_index(drop=True)
    return out


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Roh-Trades -> kanonisches Schema. Druckt die Roh-Spalten zur Kontrolle."""
    print(f"  [normalize_trades] ROH-Spalten: {list(df.columns)}")
    if df.empty:
        return pd.DataFrame(columns=["event_time", "side", "price", "quantity"])
    cols = df.columns
    c_time = _pick(cols, ["event_time", "timestamp", "time", "trade_time", "ts"])
    c_price = _pick(cols, ["price", "px"])
    c_qty = _pick(cols, ["quantity", "qty", "size", "amount"])
    c_side = _pick(cols, ["side", "aggressor", "taker_side"])
    c_maker = _pick(cols, ["is_buyer_maker", "buyer_is_maker", "maker"])
    missing = [n for n, c in [("time", c_time), ("price", c_price), ("quantity", c_qty)] if c is None]
    if missing:
        raise ValueError(
            f"Trades: konnte Spalte(n) {missing} nicht zuordnen. "
            f"Vorhandene Spalten: {list(cols)}. Bitte _pick-Kandidaten in chd_io.py ergaenzen."
        )
    if c_side is not None:
        side = _norm_side_trade(df[c_side])
    elif c_maker is not None:
        # is_buyer_maker == True  -> Kaeufer war Maker -> Aggressor ist Verkaeufer -> 'sell'
        maker = df[c_maker].astype(str).str.lower().isin(["true", "1", "yes"])
        side = np.where(maker, "sell", "buy")
    else:
        side = "unknown"
    out = pd.DataFrame({
        "event_time": _to_utc_datetime(df[c_time]),
        "side": side,
        "price": pd.to_numeric(df[c_price], errors="coerce").astype("float64"),
        "quantity": pd.to_numeric(df[c_qty], errors="coerce").astype("float64"),
    })
    out = out.dropna(subset=["event_time", "price", "quantity"])
    out = out.sort_values("event_time", kind="stable").reset_index(drop=True)
    return out


# ===========================================================================
#  Download (echte Daten) + Parquet-Cache
# ===========================================================================
def _client_kwargs() -> dict:
    key = get_api_key()
    return {"api_key": key} if key else {}


def download_orderbook() -> pd.DataFrame:
    raw = chd.get_orderbook(C.SYMBOL, C.EXCHANGE, C.START_DATE, f"{C.END_DATE} 23:59:59",
                            **_client_kwargs())
    return normalize_orderbook(raw)


def download_trades() -> pd.DataFrame:
    raw = chd.get_trades(C.SYMBOL, C.EXCHANGE, C.START_DATE, f"{C.END_DATE} 23:59:59",
                         **_client_kwargs())
    return normalize_trades(raw)


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
