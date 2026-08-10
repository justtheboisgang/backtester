"""
Orderbuch-Mechanik:

1) reconstruct_top_of_book: baut aus dem Event-Strom den besten Bid/Ask nach und
   gibt eine je-Sekunde abgetastete Reihe zurueck. Daraus schaetzen wir den
   durchschnittlichen Spread (fuer die Kosten, Anforderung 6).

2) level_deltas: pro Orderbuch-Zeile die Mengenaenderung (Delta) gegenueber dem
   vorherigen Stand desselben Preislevels. Delta>0 = Liquiditaet dazu (Stack),
   Delta<0 = Liquiditaet weg (Pull ODER Fill).

3) classify_decreases: trennt Pull (Storno) von Fill (Ausfuehrung), indem jede
   Mengenreduktion gegen den Trades-Feed abgeglichen wird (Anforderung 1).
"""

from __future__ import annotations
import heapq
import numpy as np
import pandas as pd

from . import config as C


def to_ns(s: pd.Series) -> np.ndarray:
    """
    tz-aware Zeitspalte robust in int64-Nanosekunden (seit Epoch) umwandeln.
    Noetig, weil .to_numpy() auf tz-aware Spalten je nach pandas-Version ein
    Objekt-Array liefert, das sich nicht direkt in int64 casten laesst.
    """
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")


# ---------------------------------------------------------------------------
def reconstruct_top_of_book(ob: pd.DataFrame) -> pd.DataFrame:
    """
    Rekonstruiert best_bid/best_ask aus dem Event-Strom und tastet je Sekunde ab.
    quantity ist die absolute Restmenge auf dem Level; 0 = Level entfernt.
    'snapshot'-Zeilen setzen das Buch (pro Snapshot-Zeitstempel) neu auf.

    Rueckgabe: DataFrame [event_time, best_bid, best_ask, mid, spread] (1 Zeile/Sek.)
    """
    if ob.empty:
        return pd.DataFrame(columns=["event_time", "best_bid", "best_ask", "mid", "spread"])

    et_naive = ob["event_time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    et = to_ns(ob["event_time"])
    sec = et_naive.astype("datetime64[s]")
    ev = ob["event_type"].to_numpy(dtype=object)
    sd = ob["side"].to_numpy(dtype=object)
    px = ob["price"].to_numpy(dtype="float64")
    qt = ob["quantity"].to_numpy(dtype="float64")

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    bid_heap: list[float] = []   # Max-Heap ueber -price
    ask_heap: list[float] = []   # Min-Heap ueber price

    def best_bid() -> float:
        while bid_heap:
            p = -bid_heap[0]
            if bids.get(p, 0.0) > 0.0:
                return p
            heapq.heappop(bid_heap)
        return np.nan

    def best_ask() -> float:
        while ask_heap:
            p = ask_heap[0]
            if asks.get(p, 0.0) > 0.0:
                return p
            heapq.heappop(ask_heap)
        return np.nan

    out_t, out_b, out_a = [], [], []
    cur_sec = None
    last_snap = None

    for i in range(len(et)):
        s = sec[i]
        if cur_sec is None:
            cur_sec = s
        elif s != cur_sec:
            # Zustand am Ende der vorigen Sekunde festhalten.
            out_t.append(cur_sec); out_b.append(best_bid()); out_a.append(best_ask())
            cur_sec = s

        if ev[i] == "snapshot" and et[i] != last_snap:
            bids.clear(); asks.clear(); bid_heap.clear(); ask_heap.clear()
            last_snap = et[i]

        p, q = px[i], qt[i]
        if sd[i] == "bid":
            if q > 0.0:
                if bids.get(p, 0.0) == 0.0:
                    heapq.heappush(bid_heap, -p)
                bids[p] = q
            else:
                bids.pop(p, None)
        elif sd[i] == "ask":
            if q > 0.0:
                if asks.get(p, 0.0) == 0.0:
                    heapq.heappush(ask_heap, p)
                asks[p] = q
            else:
                asks.pop(p, None)

    out_t.append(cur_sec); out_b.append(best_bid()); out_a.append(best_ask())

    tob = pd.DataFrame({
        "event_time": pd.to_datetime(out_t, utc=True),
        "best_bid": out_b,
        "best_ask": out_a,
    })
    # Ungueltige Kreuzungen (best_bid >= best_ask) verwerfen.
    tob = tob[(tob.best_ask > tob.best_bid)].reset_index(drop=True)
    tob["mid"] = (tob.best_bid + tob.best_ask) / 2.0
    tob["spread"] = tob.best_ask - tob.best_bid
    return tob


# ---------------------------------------------------------------------------
def level_deltas(ob: pd.DataFrame) -> pd.DataFrame:
    """
    Fuegt je Zeile die Mengenaenderung gegenueber dem vorigen Stand DESSELBEN
    Preislevels hinzu: Spalte 'delta' (quantity - vorherige quantity).
    Erste Beobachtung eines Levels: vorherige Menge = 0 -> delta = quantity.
    """
    if ob.empty:
        return ob.assign(delta=pd.Series(dtype="float64"))
    df = ob.sort_values("event_time", kind="stable").copy()
    prev = df.groupby(["side", "price"], sort=False)["quantity"].shift()
    df["delta"] = df["quantity"] - prev.fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Fuer BID-Level zaehlen SELL-Trades, fuer ASK-Level BUY-Trades (Aggressor = Gegenseite).
_OPP = {"bid": "sell", "ask": "buy"}


def build_fill_index(trades: pd.DataFrame) -> dict:
    """
    Baut EINMAL einen Index der Trades nach (Aggressor-Seite, gerundeter Preis)
    -> (sortierte Zeitpunkte in ns, Mengen). Wird von classify_arrays genutzt,
    damit die Klassifikation ueber viele Fenster schnell bleibt.
    """
    index: dict[tuple, tuple] = {}
    if trades.empty:
        return index
    dp = C.TRADE_MATCH_PRICE_DP
    tr = trades.copy()
    tr["price_r"] = tr["price"].round(dp)
    tr = tr.sort_values("event_time", kind="stable")
    tr = tr.assign(_t_ns=to_ns(tr["event_time"]))
    for (tside, pr), g in tr.groupby(["side", "price_r"], sort=False):
        index[(tside, round(float(pr), dp))] = (
            g["_t_ns"].to_numpy(), g["quantity"].to_numpy(dtype="float64"))
    return index


def classify_arrays(side, time_ns, price, removed, fill_index) -> np.ndarray:
    """
    Kern der Pull/Fill-Trennung auf numpy-Arrays.
    Gibt ein Array aus 'fill' / 'pull' / 'ambiguous' zurueck (Anforderung 1).
    """
    n = len(side)
    matched = np.zeros(n, dtype="float64")
    if n == 0:
        return np.empty(0, dtype=object)
    dp = C.TRADE_MATCH_PRICE_DP
    tol_ns = int(C.TRADE_MATCH_TOL_MS) * 1_000_000
    price_r = np.round(price, dp)
    for i in range(n):
        arr = fill_index.get((_OPP.get(side[i]), round(float(price_r[i]), dp)))
        if arr is None:
            continue
        times, qtys = arr
        lo = np.searchsorted(times, time_ns[i] - tol_ns, side="left")
        hi = np.searchsorted(times, time_ns[i] + tol_ns, side="right")
        if hi > lo:
            matched[i] = qtys[lo:hi].sum()
    thresh = (1.0 - C.FILL_TOLERANCE_FRAC) * removed
    return np.where(matched >= thresh, "fill",
            np.where(matched > 0.0, "ambiguous", "pull"))


def classify_decreases(dec: pd.DataFrame, fill_index: dict) -> pd.DataFrame:
    """
    Bequeme DataFrame-Huelle um classify_arrays.
    Eingabe dec: [event_time, side, price, removed_qty]; Rueckgabe: + Spalte 'klass'.
    """
    dec = dec.copy()
    if dec.empty:
        dec["klass"] = pd.Series(dtype="object")
        return dec
    klass = classify_arrays(
        dec["side"].to_numpy(dtype=object),
        to_ns(dec["event_time"]),
        dec["price"].to_numpy(dtype="float64"),
        dec["removed_qty"].to_numpy(dtype="float64"),
        fill_index,
    )
    dec["klass"] = klass
    return dec
