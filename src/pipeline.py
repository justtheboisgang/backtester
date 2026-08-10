"""
Analyse-Kern (Schritte 2-7). Alle Funktionen sind bewusst klein und einzeln
testbar. Sie arbeiten auf dem kanonischen Schema aus chd_io.

Reihenfolge:
  compute_sessions_va      -> Value Area (VAH/VAL/POC) je Session   (Schritt 2)
  build_reference_map      -> jede Session bekommt die VA der VORIGEN Session (kein Look-ahead)
  detect_events            -> Preis erreicht Referenzlevel           (Schritt 3)
  measure_events/_controls -> Pull/Stack-Imbalance vor + Move nach   (Schritt 4/5/6)
  build_report             -> Fallzahlen + Kostenvergleich           (Schritt 7)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import book

EPS = 1e-12


# ===========================================================================
#  Zeit-Helfer
# ===========================================================================
def session_bounds(date: str, start_h: int, end_h: int):
    t0 = pd.Timestamp(f"{date} 00:00:00", tz="UTC") + pd.Timedelta(hours=start_h)
    t1 = pd.Timestamp(f"{date} 00:00:00", tz="UTC") + pd.Timedelta(hours=end_h)
    return t0, t1


class PricePath:
    """
    Preis-Pfad: price_at(t) = letzter Wert <= t (kein Look-ahead).

    Standardmaessig auf dem MID-Preis aus dem rekonstruierten Orderbuch
    (price_col='mid'): der ist gegen fehlerhafte Trade-Prints (Ausreisser)
    robust. Fuer die Move-Messung ist das die saubere Quelle.
    """
    def __init__(self, df: pd.DataFrame, price_col: str = "price"):
        d = df.sort_values("event_time", kind="stable")
        d = d[d[price_col].notna() & (d[price_col] > 0)]
        self.t = book.to_ns(d["event_time"])
        self.p = d[price_col].to_numpy(dtype="float64")

    def at(self, t_ns) -> float:
        if len(self.t) == 0:
            return np.nan
        idx = np.searchsorted(self.t, t_ns, side="right") - 1
        if idx < 0:
            return np.nan
        return float(self.p[idx])


# ===========================================================================
#  Schritt 2: Value Area
# ===========================================================================
def value_area(trades_sub: pd.DataFrame, price_bin: float = None, pct: float = None) -> dict:
    """
    Volumenprofil in Preis-Bins; POC = Bin mit meistem Volumen; Value Area waechst
    vom POC aus (jeweils die volumenstaerkere Nachbarseite), bis pct des Volumens
    abgedeckt ist. Rueckgabe: poc/vah/val (Bin-Mitten) + total_vol + n_trades.
    """
    price_bin = price_bin or C.VA_PRICE_BIN
    pct = pct or C.VALUE_AREA_PCT
    if trades_sub.empty:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "total_vol": 0.0, "n_trades": 0}

    bin_id = np.floor(trades_sub["price"].to_numpy() / price_bin).astype("int64")
    vol = trades_sub["quantity"].to_numpy(dtype="float64")
    prof = pd.Series(vol).groupby(bin_id).sum().sort_index()
    ids = prof.index.to_numpy()
    vols = prof.to_numpy()
    total = vols.sum()

    poc_pos = int(np.argmax(vols))
    lo = hi = poc_pos
    covered = vols[poc_pos]
    target = pct * total
    while covered < target and (lo > 0 or hi < len(vols) - 1):
        up_v = vols[hi + 1] if hi < len(vols) - 1 else -1.0
        dn_v = vols[lo - 1] if lo > 0 else -1.0
        if up_v >= dn_v:
            hi += 1; covered += vols[hi]
        else:
            lo -= 1; covered += vols[lo]

    def center(bin_index_value):
        return (bin_index_value + 0.5) * price_bin

    return {
        "poc": center(ids[poc_pos]),
        "vah": center(ids[hi]),      # oberer Rand der Value Area
        "val": center(ids[lo]),      # unterer Rand der Value Area
        "total_vol": float(total),
        "n_trades": int(len(trades_sub)),
    }


def compute_sessions_va(trades: pd.DataFrame, date: str = None) -> pd.DataFrame:
    """Value Area je Session am Tag `date`."""
    date = date or C.START_DATE
    rows = []
    for name, sh, eh in C.SESSIONS:
        t0, t1 = session_bounds(date, sh, eh)
        sub = trades[(trades.event_time >= t0) & (trades.event_time < t1)]
        va = value_area(sub)
        rows.append({"session": name, "start": t0, "end": t1, **va})
    return pd.DataFrame(rows)


def build_reference_map(va_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kein Look-ahead: die VA einer Session dient als Referenz fuer die FOLGENDE.
    Am ersten Tag hat die erste Session (Asia) keinen Vorgaenger -> entfaellt.
    Rueckgabe: je Ziel-Session [session, start, end, ref_session, ref_vah, ref_val].
    """
    va_df = va_df.reset_index(drop=True)
    rows = []
    for i in range(1, len(va_df)):
        prev, cur = va_df.iloc[i - 1], va_df.iloc[i]
        rows.append({
            "session": cur["session"], "start": cur["start"], "end": cur["end"],
            "ref_session": prev["session"],
            "ref_vah": prev["vah"], "ref_val": prev["val"],
        })
    return pd.DataFrame(rows)


# ===========================================================================
#  Schritt 3: Events (Preis erreicht Referenzlevel)
# ===========================================================================
def _touch_events(trades_sub, level_price, level_type, session, ref_session):
    """Ein Event, sobald der Preis das Band um `level_price` betritt (mit Abklingzeit)."""
    if np.isnan(level_price) or trades_sub.empty:
        return []
    t = trades_sub["event_time"].to_numpy()
    p = trades_sub["price"].to_numpy(dtype="float64")
    band = C.LEVEL_BAND
    cooldown = np.timedelta64(int(C.EVENT_COOLDOWN_S), "s")
    events = []
    armed = True
    last_event_t = None
    for i in range(len(t)):
        inside = abs(p[i] - level_price) <= band
        if inside and armed and (last_event_t is None or (t[i] - last_event_t) >= cooldown):
            events.append({
                "t0": pd.Timestamp(t[i]), "level_type": level_type,
                "level_price": float(level_price),
                "session": session, "ref_session": ref_session,
            })
            armed = False
            last_event_t = t[i]
        elif not inside:
            armed = True   # Preis hat das Band verlassen -> wieder scharf
    return events


def detect_events(trades: pd.DataFrame, ref_map: pd.DataFrame) -> pd.DataFrame:
    """Alle Events ueber alle Ziel-Sessions (VAH und VAL der Vor-Session)."""
    all_events = []
    for _, r in ref_map.iterrows():
        sub = trades[(trades.event_time >= r["start"]) & (trades.event_time < r["end"])]
        all_events += _touch_events(sub, r["ref_vah"], "VAH", r["session"], r["ref_session"])
        all_events += _touch_events(sub, r["ref_val"], "VAL", r["session"], r["ref_session"])
    cols = ["t0", "level_type", "level_price", "session", "ref_session"]
    if not all_events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(all_events).sort_values("t0").reset_index(drop=True)[cols]


# ===========================================================================
#  Schritt 4/5: Messung Imbalance (vorher) + Outcome (nachher)
# ===========================================================================
class Measurer:
    """Buendelt die Arrays, die fuer die Fenster-Messungen gebraucht werden."""
    def __init__(self, ob_delta: pd.DataFrame, trades: pd.DataFrame, price_path: PricePath):
        obd = ob_delta.sort_values("event_time", kind="stable")
        self.ob_t = book.to_ns(obd["event_time"])
        self.ob_side = obd["side"].to_numpy(dtype=object)
        self.ob_px = obd["price"].to_numpy(dtype="float64")
        self.ob_delta = obd["delta"].to_numpy(dtype="float64")
        self.fill_index = book.build_fill_index(trades)
        self.pp = price_path

    def imbalance(self, center: float, t0_ns: int, pre_s: int) -> dict:
        """Pull/Stack-Imbalance im Band um `center` im Fenster [t0-pre, t0]."""
        pre_ns = int(pre_s) * 1_000_000_000
        lo = np.searchsorted(self.ob_t, t0_ns - pre_ns, side="left")
        hi = np.searchsorted(self.ob_t, t0_ns, side="right")
        base = {"imbalance": 0.0, "stack_bid": 0.0, "stack_ask": 0.0,
                "pull_bid": 0.0, "pull_ask": 0.0, "fill_vol": 0.0,
                "n_dec": 0, "n_amb": 0}
        if hi <= lo:
            return base
        side = self.ob_side[lo:hi]; px = self.ob_px[lo:hi]
        delta = self.ob_delta[lo:hi]; t = self.ob_t[lo:hi]
        band = np.abs(px - center) <= C.LEVEL_BAND
        if not band.any():
            return base
        side, px, delta, t = side[band], px[band], delta[band], t[band]

        inc = delta > 0
        stack_bid = delta[inc & (side == "bid")].sum()
        stack_ask = delta[inc & (side == "ask")].sum()

        dec = delta < 0
        pull_bid = pull_ask = fill_vol = 0.0
        n_dec = int(dec.sum()); n_amb = 0
        if n_dec:
            klass = book.classify_arrays(side[dec], t[dec], px[dec], -delta[dec], self.fill_index)
            rem = -delta[dec]; dside = side[dec]
            is_pull = klass == "pull"
            pull_bid = rem[is_pull & (dside == "bid")].sum()
            pull_ask = rem[is_pull & (dside == "ask")].sum()
            fill_vol = rem[klass == "fill"].sum()
            n_amb = int((klass == "ambiguous").sum())

        net_bid = stack_bid - pull_bid
        net_ask = stack_ask - pull_ask
        total = stack_bid + stack_ask + pull_bid + pull_ask
        imb = (net_bid - net_ask) / total if total > EPS else 0.0
        return {"imbalance": float(imb), "stack_bid": float(stack_bid),
                "stack_ask": float(stack_ask), "pull_bid": float(pull_bid),
                "pull_ask": float(pull_ask), "fill_vol": float(fill_vol),
                "n_dec": n_dec, "n_amb": n_amb}

    def outcome(self, t0_ns: int, post_s: int) -> dict:
        """Signierter Preis-Move ueber das Nach-Fenster."""
        p0 = self.pp.at(t0_ns)
        p1 = self.pp.at(t0_ns + int(post_s) * 1_000_000_000)
        if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
            return {"p0": p0, "p1": p1, "move_bps": np.nan, "abs_move_bps": np.nan}
        move_bps = (p1 / p0 - 1.0) * 1e4
        return {"p0": float(p0), "p1": float(p1),
                "move_bps": float(move_bps), "abs_move_bps": float(abs(move_bps))}


def _measure_points(points: pd.DataFrame, meas: Measurer, group: str) -> pd.DataFrame:
    """
    points: [t0, center]  (+ optionale Zusatzspalten, die durchgereicht werden).
    Fuer JEDE Kombination (pre_window x post_horizon) eine Zeile.
    """
    extra_cols = [c for c in points.columns if c not in ("t0", "center")]
    out = []
    for _, r in points.iterrows():
        t0_ns = int(pd.Timestamp(r["t0"]).value)
        center = float(r["center"])
        base = {c: r[c] for c in extra_cols}
        base.update({"group": group, "t0": pd.Timestamp(r["t0"]), "center": center})
        for pre_s in C.PRE_WINDOWS_S:
            imb = meas.imbalance(center, t0_ns, pre_s)
            for post_s in C.POST_HORIZONS_S:
                oc = meas.outcome(t0_ns, post_s)
                row = dict(base)
                row["pre_s"] = pre_s
                row["post_s"] = post_s
                row.update(imb)
                row.update(oc)
                out.append(row)
    return pd.DataFrame(out)


def measure_events(events: pd.DataFrame, meas: Measurer) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    pts = events.rename(columns={"level_price": "center"}).copy()
    pts["center_kind"] = pts["level_type"]
    return _measure_points(pts[["t0", "center", "level_type", "session", "ref_session"]],
                           meas, group="event")


# ===========================================================================
#  Schritt 6: Kontrollgruppe (zufaellige Zeitpunkte OHNE Levelbezug)
# ===========================================================================
def build_controls(trades, ref_map, meas: Measurer, n=None, seed=None) -> pd.DataFrame:
    """
    Zieht zufaellige Zeitpunkte im Handelszeitraum, deren aktueller Preis
    weit genug (CONTROL_MIN_DIST) von JEDEM Referenzlevel entfernt ist, und
    misst dort dieselbe Groesse. Ohne diese Kontrolle misst man nur den
    allgemeinen Marktrhythmus (Anforderung 4).
    """
    n = n or C.N_CONTROL
    rng = np.random.default_rng(seed if seed is not None else C.RANDOM_SEED)
    if trades.empty or ref_map.empty:
        return pd.DataFrame()
    t_lo = int(trades.event_time.min().value)
    t_hi = int(trades.event_time.max().value)
    levels = []
    for _, r in ref_map.iterrows():
        for lv in (r["ref_vah"], r["ref_val"]):
            if not np.isnan(lv):
                levels.append(lv)
    levels = np.array(levels)

    pts = []
    tries = 0
    max_tries = n * 50
    while len(pts) < n and tries < max_tries:
        tries += 1
        t_ns = int(rng.integers(t_lo, t_hi))
        price = meas.pp.at(t_ns)
        if np.isnan(price):
            continue
        if levels.size and np.min(np.abs(levels - price)) < C.CONTROL_MIN_DIST:
            continue   # zu nah an einem Level -> kein sauberer Kontrollpunkt
        pts.append({"t0": pd.Timestamp(t_ns, tz="UTC"), "center": float(price)})
    if not pts:
        return pd.DataFrame()
    return _measure_points(pd.DataFrame(pts), meas, group="control")


# ===========================================================================
#  Schritt 7: Report (Fallzahlen + Kostenvergleich + Multiple-Testing)
# ===========================================================================
def mean_spread_bps(tob: pd.DataFrame) -> float:
    if tob.empty:
        return np.nan
    rel = (tob["spread"] / tob["mid"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(rel.mean() * 1e4) if len(rel) else np.nan


def build_report(ev_meas: pd.DataFrame, ct_meas: pd.DataFrame, tob: pd.DataFrame) -> pd.DataFrame:
    """
    Ein Ergebnis je (pre_window x post_horizon):
      Fallzahlen (Flag <30 = nicht belastbar), mittlere Imbalance Event vs Kontrolle,
      Richtungs-Zusammenhang (Korrelation Imbalance<->Move), mittlerer |Move| vs Kosten.
    """
    spread_bps = mean_spread_bps(tob)
    cost_bps = C.ROUND_TRIP_FEE_BPS + (spread_bps if not np.isnan(spread_bps) else 0.0)

    combos = [(pre, post) for pre in C.PRE_WINDOWS_S for post in C.POST_HORIZONS_S]
    n_tests = len(combos)  # Anzahl getesteter Varianten (Anforderung 7)

    rows = []
    for pre, post in combos:
        ev = ev_meas[(ev_meas.pre_s == pre) & (ev_meas.post_s == post)] if len(ev_meas) else ev_meas
        ct = ct_meas[(ct_meas.pre_s == pre) & (ct_meas.post_s == post)] if len(ct_meas) else ct_meas
        n_ev = len(ev); n_ct = len(ct)

        def corr(df):
            if len(df) < 3:
                return np.nan
            d = df[["imbalance", "move_bps"]].dropna()
            if len(d) < 3 or d["imbalance"].std() < EPS or d["move_bps"].std() < EPS:
                return np.nan
            return float(np.corrcoef(d["imbalance"], d["move_bps"])[0, 1])

        ev_abs = float(ev["abs_move_bps"].mean()) if n_ev else np.nan
        ct_abs = float(ct["abs_move_bps"].mean()) if n_ct else np.nan
        rows.append({
            "pre_s": pre, "post_s": post,
            "n_events": n_ev, "n_control": n_ct,
            "belastbar": (n_ev >= 30 and n_ct >= 30),
            "imb_event_mean": float(ev["imbalance"].mean()) if n_ev else np.nan,
            "imb_control_mean": float(ct["imbalance"].mean()) if n_ct else np.nan,
            "corr_imb_move_event": corr(ev),
            "abs_move_event_bps": ev_abs,
            "abs_move_control_bps": ct_abs,
            "spread_bps": spread_bps,
            "cost_bps": cost_bps,
            "net_event_bps": (ev_abs - cost_bps) if not np.isnan(ev_abs) else np.nan,
            "handelbar": (not np.isnan(ev_abs)) and (ev_abs > cost_bps),
            "n_tests_total": n_tests,
        })
    return pd.DataFrame(rows)
