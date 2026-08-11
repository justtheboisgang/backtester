"""
DOM-Studie: Sagt der Orderbuch-Zustand VOR t0 eine groessere Bewegung voraus -
und deren Richtung? Zeithorizonte 30/60/120 s.

Ablauf pro Tag (memory-sicher; Rohdaten nur transient):
  1) build_second_table : je Sekunde des Tages ein Zustand des Buchs
     (mid, spread, Tiefe) + Fluss (stack/pull je Seite, Volumen, Trade-Imbalance).
  2) build_sample       : je Sekunde t0 die Vor-Fenster-Merkmale (5/15/30 s,
     ausschliesslich VOR t0 -> kein Look-ahead) und die Vorwaertsbewegung
     (30/60/120 s). Plus Flags near_level und active.
Die kleine Sample-Tabelle (je Sekunde eine Zeile) wird als Parquet behalten,
die Rohdaten werden verworfen. Analysen laufen ueber diese Tabellen.

Vorzeichen-Konvention imbalance:
  imbalance = ((stack_bid + pull_ask) - (stack_ask + pull_bid)) / Summe
  > 0  = Aufwaertsdruck  (Bids aufgebaut / Asks gezogen)
  < 0  = Abwaertsdruck   (Asks aufgebaut / Bids gezogen)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import book
from . import pipeline as P

EPS = 1e-12
N_SEC = 24 * 3600


# ===========================================================================
#  1) Sekundentabelle
# ===========================================================================
def _bincount(idx, weights, n=N_SEC):
    if len(idx) == 0:
        return np.zeros(n, dtype="float64")
    return np.bincount(idx, weights=weights, minlength=n)[:n]


def build_second_table(ob: pd.DataFrame, trades: pd.DataFrame, date: str) -> pd.DataFrame:
    """Je Sekunde des Tages: Buchzustand (mid/spread/Tiefe) + Fluss (stack/pull/Volumen)."""
    day0_ns = pd.Timestamp(f"{date} 00:00:00", tz="UTC").value

    # Buchzustand je Sekunde (dicht, inkl. Tiefe).
    bsec = book.top_of_book_by_second(ob, date)          # mid, spread, bid_qty, ask_qty
    mid_by_sec = bsec["mid"].to_numpy()

    # --- Orderbuch-Fluss nahe dem Best (Band um den Mid) ---
    obd = book.level_deltas(ob)
    ns = book.to_ns(obd["event_time"])
    sec = ((ns - day0_ns) // 1_000_000_000).astype("int64")
    side = obd["side"].to_numpy(dtype=object)
    price = obd["price"].to_numpy(dtype="float64")
    delta = obd["delta"].to_numpy(dtype="float64")

    valid = (sec >= 0) & (sec < N_SEC)
    mid_at = np.full(len(sec), np.nan)
    mid_at[valid] = mid_by_sec[sec[valid]]
    keep = valid & np.isfinite(mid_at) & (np.abs(price - mid_at) <= C.DOM_BOOK_BAND)

    ks, kside, kdelta, kns, kprice = sec[keep], side[keep], delta[keep], ns[keep], price[keep]
    inc = kdelta > 0
    is_bid = kside == "bid"
    is_ask = kside == "ask"

    stack_bid = _bincount(ks[inc & is_bid], kdelta[inc & is_bid])
    stack_ask = _bincount(ks[inc & is_ask], kdelta[inc & is_ask])

    # Mengenreduktionen: Pull vs. Fill (einmal pro Tag klassifizieren).
    dec = kdelta < 0
    fill_index = book.build_fill_index(trades)
    klass = book.classify_arrays(kside[dec], kns[dec], kprice[dec], -kdelta[dec], fill_index)
    drem = -kdelta[dec]; dsec = ks[dec]; dside = kside[dec]
    is_pull = klass == "pull"
    pull_bid = _bincount(dsec[is_pull & (dside == "bid")], drem[is_pull & (dside == "bid")])
    pull_ask = _bincount(dsec[is_pull & (dside == "ask")], drem[is_pull & (dside == "ask")])
    n_dec = _bincount(dsec, np.ones(dsec.shape))
    n_amb = _bincount(dsec[klass == "ambiguous"], np.ones((klass == "ambiguous").sum()))

    # --- Trades je Sekunde ---
    tns = book.to_ns(trades["event_time"])
    tsec = ((tns - day0_ns) // 1_000_000_000).astype("int64")
    tvalid = (tsec >= 0) & (tsec < N_SEC)
    tqty = trades["quantity"].to_numpy(dtype="float64")
    tbuy = (trades["side"].to_numpy(dtype=object) == "buy")
    buy_vol = _bincount(tsec[tvalid & tbuy], tqty[tvalid & tbuy])
    sell_vol = _bincount(tsec[tvalid & ~tbuy], tqty[tvalid & ~tbuy])

    st = bsec.copy()
    st["stack_bid"] = stack_bid
    st["stack_ask"] = stack_ask
    st["pull_bid"] = pull_bid
    st["pull_ask"] = pull_ask
    st["buy_vol"] = buy_vol
    st["sell_vol"] = sell_vol
    st["trade_vol"] = buy_vol + sell_vol
    st["n_dec"] = n_dec
    st["n_amb"] = n_amb
    return st


# ===========================================================================
#  2) Sample-Tabelle (Merkmale vor t0 + Vorwaertsbewegung)
# ===========================================================================
def _ref_levels_by_sec(trades: pd.DataFrame, date: str):
    """VAH/VAL der jeweiligen Vor-Session, je Sekunde (NaN wo keine Referenz)."""
    day0_ns = pd.Timestamp(f"{date} 00:00:00", tz="UTC").value
    va = P.compute_sessions_va(trades, date)
    ref = P.build_reference_map(va)
    vah = np.full(N_SEC, np.nan); val = np.full(N_SEC, np.nan)
    for _, r in ref.iterrows():
        s0 = int((r["start"].value - day0_ns) // 1_000_000_000)
        s1 = int((r["end"].value - day0_ns) // 1_000_000_000)
        s0, s1 = max(0, s0), min(N_SEC, s1)
        vah[s0:s1] = r["ref_vah"]; val[s0:s1] = r["ref_val"]
    return vah, val


def build_sample(st: pd.DataFrame, trades: pd.DataFrame, date: str) -> pd.DataFrame:
    """Je Sekunde t0: Vor-Fenster-Merkmale + Vorwaertsbewegung. Kein Look-ahead."""
    mid = st["mid"].to_numpy()
    spread = st["spread"].to_numpy()
    bid_qty = st["bid_qty"].to_numpy()
    ask_qty = st["ask_qty"].to_numpy()

    flow = {k: st[k].to_numpy() for k in
            ["stack_bid", "stack_ask", "pull_bid", "pull_ask", "buy_vol", "sell_vol", "trade_vol"]}
    # Prefix-Summen fuer schnelle Fenstersummen: cum[k][i] = Summe der Sekunden 0..i-1.
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in flow.items()}

    w_max = max(C.PRE_WINDOWS_S)
    h_max = max(C.HORIZONS_S)
    t0 = np.arange(w_max, N_SEC - h_max, dtype="int64")   # gueltige Zeitpunkte

    def wsum(k, w):
        # Summe der Sekunden [t0-w, t0-1]  (streng VOR t0) = cum[t0] - cum[t0-w]
        return cum[k][t0] - cum[k][t0 - w]

    out = {"date": date, "t0_sec": t0}
    # Zustand direkt vor t0 (Sekunde t0-1): Spread & Tiefe.
    out["spread"] = spread[t0 - 1]
    out["bid_depth"] = bid_qty[t0 - 1]
    out["ask_depth"] = ask_qty[t0 - 1]
    out["depth"] = bid_qty[t0 - 1] + ask_qty[t0 - 1]
    out["mid_t0"] = mid[t0]

    for w in C.PRE_WINDOWS_S:
        sb, sa = wsum("stack_bid", w), wsum("stack_ask", w)
        pb, pa = wsum("pull_bid", w), wsum("pull_ask", w)
        bv, sv = wsum("buy_vol", w), wsum("sell_vol", w)
        out[f"stack_bid_{w}"] = sb
        out[f"stack_ask_{w}"] = sa
        out[f"pull_bid_{w}"] = pb
        out[f"pull_ask_{w}"] = pa
        bull = sb + pa; bear = sa + pb
        out[f"imbalance_{w}"] = (bull - bear) / (bull + bear + EPS)
        out[f"vol_{w}"] = bv + sv
        out[f"trade_imb_{w}"] = (bv - sv) / (bv + sv + EPS)

    # Vorwaertsbewegung je Horizont (auf dem Mid).
    for h in C.HORIZONS_S:
        m1 = mid[t0 + h]
        move = (m1 / mid[t0] - 1.0) * 1e4
        out[f"move_{h}"] = move
        out[f"absmove_{h}"] = np.abs(move)

    # Flags.
    vah, val = _ref_levels_by_sec(trades, date)
    near = (np.abs(mid[t0] - vah[t0]) <= C.DOM_NEAR_LEVEL_BAND) | \
           (np.abs(mid[t0] - val[t0]) <= C.DOM_NEAR_LEVEL_BAND)
    out["near_level"] = np.where(np.isfinite(near), near, False)
    out["active"] = wsum("trade_vol", w_max) > 0     # aktive Phase = Handel in letzten 30 s

    df = pd.DataFrame(out)
    # Zeitpunkte ohne gueltigen Buchzustand verwerfen (z. B. vor erstem Snapshot).
    df = df[np.isfinite(df["mid_t0"]) & (df["mid_t0"] > 0)]
    for h in C.HORIZONS_S:
        df = df[np.isfinite(df[f"move_{h}"])]
    return df.reset_index(drop=True)


# ===========================================================================
#  3) Analysen
# ===========================================================================
# Kuratierte Vorher-Merkmale fuer den Groessen-Vergleich (je Fenster).
_WIN_FEATURES = ["imbalance", "trade_imb", "stack_bid", "stack_ask", "pull_bid", "pull_ask", "vol"]
_STATE_FEATURES = ["spread", "depth"]   # fensterunabhaengig (Zustand bei t0)

_THRESHOLDS = [("taker_12bps", C.THRESH_TAKER_BPS), ("maker_6bps", C.THRESH_MAKER_BPS)]


def move_distribution(sample: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Kernfrage: Verteilung der |move| je Horizont + Anteil ueber den Schwellen."""
    rows = []
    for h in C.HORIZONS_S:
        a = sample[f"absmove_{h}"].dropna()
        n = len(a)
        rows.append({
            "scope": scope, "horizon_s": h, "n": n,
            "median": a.median() if n else np.nan,
            "p75": a.quantile(0.75) if n else np.nan,
            "p90": a.quantile(0.90) if n else np.nan,
            "p95": a.quantile(0.95) if n else np.nan,
            "p99": a.quantile(0.99) if n else np.nan,
            "max": a.max() if n else np.nan,
            "frac_gt_6bps": float((a > C.THRESH_MAKER_BPS).mean()) if n else np.nan,
            "frac_gt_12bps": float((a > C.THRESH_TAKER_BPS).mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def size_analysis(sample: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Unterscheiden sich die Vorher-Merkmale zwischen grossen und kleinen Folge-Moves?"""
    rows = []
    for w in C.PRE_WINDOWS_S:
        for h in C.HORIZONS_S:
            for thr_name, thr in _THRESHOLDS:
                big = sample[f"absmove_{h}"] > thr
                g_big, g_small = sample[big], sample[~big]
                n_big, n_small = len(g_big), len(g_small)
                rec = {"scope": scope, "pre_s": w, "horizon_s": h, "threshold": thr_name,
                       "n_big": n_big, "n_small": n_small,
                       "belastbar": (n_big >= 30 and n_small >= 30)}
                for f in _WIN_FEATURES:
                    col = f"{f}_{w}"
                    rec[f"{f}__big"] = float(g_big[col].mean()) if n_big else np.nan
                    rec[f"{f}__small"] = float(g_small[col].mean()) if n_small else np.nan
                for f in _STATE_FEATURES:
                    rec[f"{f}__big"] = float(g_big[f].mean()) if n_big else np.nan
                    rec[f"{f}__small"] = float(g_small[f].mean()) if n_small else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def direction_analysis(sample: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    Nur unter den grossen Moves: sagt die Vorher-Imbalance das Vorzeichen vorher?

    'base_up' = Anteil der grossen Moves, die nach OBEN gingen. Das ist die
    Basisrate: waere sie z. B. 0.60, erreicht schon 'immer long' 60 % Treffer.
    Eine Trefferquote ist nur relativ zu max(base_up, 1-base_up) interpretierbar.
    """
    rows = []
    for w in C.PRE_WINDOWS_S:
        for h in C.HORIZONS_S:
            for thr_name, thr in _THRESHOLDS:
                big = sample[sample[f"absmove_{h}"] > thr]
                n_big = len(big)
                move_sign = np.sign(big[f"move_{h}"].to_numpy())
                base_up = float((move_sign > 0).mean()) if n_big else np.nan
                rec = {"scope": scope, "pre_s": w, "horizon_s": h, "threshold": thr_name,
                       "n_big": n_big, "belastbar": (n_big >= 30),
                       "base_up": base_up,
                       "base_rate": max(base_up, 1.0 - base_up) if n_big else np.nan}
                for sig in ["imbalance", "trade_imb"]:
                    x = big[f"{sig}_{w}"].to_numpy()
                    use = np.abs(x) > EPS
                    n_use = int(use.sum())
                    hit = float((np.sign(x[use]) == move_sign[use]).mean()) if n_use else np.nan
                    rec[f"hit_{sig}"] = hit
                    rec[f"n_{sig}"] = n_use
                rows.append(rec)
    return pd.DataFrame(rows)


def build_control(sample: pd.DataFrame, n: int = None, seed: int = None) -> pd.DataFrame:
    """
    Kontrollgruppe: zufaellige Zeitpunkte aus AKTIVEN Phasen, gleiche Messung.
    Dient als Referenz - ohne sie misst man nur den allgemeinen Marktrhythmus.
    """
    n = n or C.N_CONTROL
    act = sample[sample["active"]] if "active" in sample.columns else sample
    if act.empty:
        return act
    rng = np.random.default_rng(seed if seed is not None else C.RANDOM_SEED)
    take = min(n, len(act))
    idx = rng.choice(len(act), size=take, replace=False)
    return act.iloc[np.sort(idx)].reset_index(drop=True)


def n_tested_variants() -> dict:
    """Anzahl getesteter Varianten (Anforderung: Multiple Testing ausweisen)."""
    w, h, thr = len(C.PRE_WINDOWS_S), len(C.HORIZONS_S), len(_THRESHOLDS)
    return {
        "verteilung_horizonte": h,
        "groesse_zellen": w * h * thr,               # je Zelle mehrere Merkmale verglichen
        "groesse_merkmale_pro_zelle": len(_WIN_FEATURES) + len(_STATE_FEATURES),
        "richtung_zellen": w * h * thr,
        "richtung_signale_pro_zelle": 2,             # imbalance & trade_imb
    }
