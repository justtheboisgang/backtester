"""
Synthetischer Daten-Generator (nur zum OFFLINE-Testen der Pipeline).

Erzeugt Orderbuch- und Trade-Daten im KANONISCHEN Schema, damit die
Methodik (Schritte 2-7) ohne Netzwerk und ohne API-Key end-to-end laufen kann.
Die Daten sind bewusst kuenstlich - sie beweisen, dass die LOGIK sauber
durchlaeuft, NICHT dass ein Effekt existiert.

Enthaelt beide Faelle, die die Pull/Fill-Trennung braucht:
  - Mengenreduktionen MIT gleichzeitigem Trade  -> sollten als 'fill' erkannt werden
  - Mengenreduktionen OHNE Trade                -> sollten als 'pull' erkannt werden
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def make_synthetic(date: str = None, hours: int = 24, seed: int = 0):
    date = date or C.START_DATE
    rng = np.random.default_rng(seed)
    day0 = pd.Timestamp(f"{date} 00:00:00", tz="UTC")
    n_sec = hours * 3600

    # --- Mid-Preis pro Sekunde: Session-Zentren + langsame Oszillation ---
    price = 150.0
    mids = np.empty(n_sec, dtype="float64")
    for s in range(n_sec):
        hour = s // 3600
        center = 150.0 if hour < 8 else (151.0 if hour < 13 else 149.5)
        price += rng.normal(0, 0.015) + (center - price) * 0.001
        osc = 0.5 * np.sin(2 * np.pi * s / 1800.0)   # ~30-min-Welle, Amplitude 0.5
        mids[s] = price + osc

    tick = 0.01
    depth = 6
    ob_rows = []
    tr_rows = []

    def bid_levels(mp):
        return {round(mp - i * tick, 2) for i in range(1, depth + 1)}

    def ask_levels(mp):
        return {round(mp + i * tick, 2) for i in range(1, depth + 1)}

    # --- Anfangs-Snapshot (setzt das Buch auf) ---
    mp0 = round(mids[0], 2)
    cur_bids = bid_levels(mp0)
    cur_asks = ask_levels(mp0)
    for p in cur_bids:
        ob_rows.append((day0, "snapshot", "bid", p, round(20 + rng.random() * 80, 2)))
    for p in cur_asks:
        ob_rows.append((day0, "snapshot", "ask", p, round(20 + rng.random() * 80, 2)))

    # --- Pro Sekunde: kohaerentes Buch pflegen + Trades ---
    for s in range(n_sec):
        t = day0 + pd.Timedelta(seconds=s)
        m = mids[s]
        mp = round(m, 2)
        tgt_bids = bid_levels(mp)
        tgt_asks = ask_levels(mp)

        # PULLs durch Marktbewegung: Levels, die nicht mehr aktuell sind, loeschen (qty 0, KEIN Trade)
        for p in cur_bids - tgt_bids:
            ob_rows.append((t, "update", "bid", p, 0.0))
        for p in cur_asks - tgt_asks:
            ob_rows.append((t, "update", "ask", p, 0.0))
        # neue Levels dazu (Stack)
        for p in tgt_bids - cur_bids:
            ob_rows.append((t, "update", "bid", p, round(20 + rng.random() * 80, 2)))
        for p in tgt_asks - cur_asks:
            ob_rows.append((t, "update", "ask", p, round(20 + rng.random() * 80, 2)))
        cur_bids, cur_asks = tgt_bids, tgt_asks

        # STACK: Menge auf einem bestehenden Level erhoehen
        if rng.random() < 0.5:
            p = round(mp - rng.integers(1, depth + 1) * tick, 2)
            ob_rows.append((t, "update", "bid", p, round(50 + rng.random() * 80, 2)))
        # reiner PULL: Menge auf einem Level reduzieren, OHNE Trade
        if rng.random() < 0.5:
            p = round(mp + rng.integers(1, depth + 1) * tick, 2)
            ob_rows.append((t + pd.Timedelta(milliseconds=500), "update", "ask", p,
                            round(5 + rng.random() * 5, 2)))

        # 1-2 Trades pro Sekunde -> FILL am jeweiligen Best-Level (gleiche Zeit, gleicher Preis)
        n_tr = rng.integers(1, 3)
        for _ in range(n_tr):
            if rng.random() < 0.5:
                side, bside, px = "buy", "ask", round(mp + tick, 2)   # hebt den Ask
            else:
                side, bside, px = "sell", "bid", round(mp - tick, 2)  # trifft den Bid
            qty = float(round(1 + rng.random() * 8, 2))
            t_tr = t + pd.Timedelta(milliseconds=int(rng.integers(0, 999)))
            tr_rows.append((t_tr, side, px, qty))
            ob_rows.append((t_tr, "update", bside, px, round(max(1.0, 50.0 - qty), 2)))

    ob = pd.DataFrame(ob_rows, columns=["event_time", "event_type", "side", "price", "quantity"])
    ob = ob.sort_values("event_time", kind="stable").reset_index(drop=True)
    tr = pd.DataFrame(tr_rows, columns=["event_time", "side", "price", "quantity"])
    tr = tr.sort_values("event_time", kind="stable").reset_index(drop=True)
    return ob, tr
