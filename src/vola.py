"""
Volatilitaets-Studie: Sagt der Orderbuch-Zustand die realisierte Volatilitaet
der naechsten 30/60/120 s vorher - ueber vergangene Volatilitaet hinaus?

Zielgroessen (beide ausgewiesen), gemessen NACH t0:
  rv_<h>    : Standardabweichung der Sekunden-Returns ueber den Horizont (bps)
  range_<h> : High-Low-Spanne des Mid ueber den Horizont, relativ zu mid[t0] (bps)

Merkmale VOR t0 (Fenster 5/15/30 s), einzeln ausgewiesen:
  rv_pre_<w>       vergangene realisierte Volatilitaet  (die Basis)
  netout_<w>       Netto-Abfluss (pull_bid+pull_ask) - (stack_bid+stack_ask)
  depth_ratio_<w>  Tiefe am Fensterende / Tiefe am Fensteranfang
  churn_<w>        Umschichtungsrate (pull + stack) / Tiefe
  vol_<w>          gehandeltes Volumen
  ntrades_<w>      Trade-Anzahl
  depth, spread_bps  Zustand bei t0 (fensterunabhaengig)

KEIN LOOK-AHEAD: Alle Fenstersummen laufen ueber Sekunden <= t0; Returns im
Vorfenster nutzen nur Mid-Werte bis t0. Zielgroessen ausschliesslich nach t0.

DIE TREPPE (genestet, jeweils out-of-sample bewertet):
  Stufe 1: nur vergangene Volatilitaet
  Stufe 2: + gehandeltes Volumen und Trade-Anzahl
  Stufe 3: + Orderbuch-Merkmale (Netto-Abfluss, Tiefenveraenderung, Umschichtung,
           Tiefe, Spread)
Nur wenn Stufe 3 die Stufe 2 OUT-OF-SAMPLE schlaegt, traegt das Orderbuch eigene
Information. In-sample steigt R2 durch zusaetzliche Merkmale IMMER - deshalb wird
mit Leave-one-day-out-Kreuzvalidierung bewertet (trainieren auf 4 Tagen, testen
auf dem 5.).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

EPS = 1e-12
N_SEC = 24 * 3600

# Merkmalsgruppen der Treppe (je Fenster w eingesetzt).
STAGE1 = ["log_rv_pre"]
STAGE2 = STAGE1 + ["log_vol", "log_ntrades"]
STAGE3 = STAGE2 + ["netout_norm", "depth_ratio", "churn", "log_depth", "spread_bps"]
STAGES = {"S1_nur_Vola": STAGE1, "S2_plus_Volumen": STAGE2, "S3_plus_Orderbuch": STAGE3}

TARGETS = ["rv", "range"]


# ===========================================================================
#  Sample-Bau
# ===========================================================================
def build_vola_sample(st: pd.DataFrame, date: str) -> pd.DataFrame:
    """Aus der Sekundentabelle die Merkmale vor t0 und die Ziele nach t0 bauen."""
    mid = st["mid"].to_numpy(dtype="float64")
    n = len(mid)

    # Sekunden-Returns in bps: r[s] = ln(mid[s]/mid[s-1]) * 1e4. r[0] = 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.zeros(n)
        r[1:] = np.log(mid[1:] / mid[:-1]) * 1e4
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # Prefix-Summen fuer schnelle Fenster-Statistiken.
    cs = np.concatenate([[0.0], np.cumsum(r)])
    cs2 = np.concatenate([[0.0], np.cumsum(r * r)])

    def rv(a, b):
        """Std der Returns ueber die Sekunden [a, b] (inklusiv), elementweise."""
        cnt = (b - a + 1).astype("float64")
        s1 = cs[b + 1] - cs[a]
        s2 = cs2[b + 1] - cs2[a]
        var = s2 / cnt - (s1 / cnt) ** 2
        return np.sqrt(np.maximum(var, 0.0))

    flow_cols = ["stack_bid", "stack_ask", "pull_bid", "pull_ask", "trade_vol", "n_trades"]
    cum = {k: np.concatenate([[0.0], np.cumsum(st[k].to_numpy(dtype="float64"))])
           for k in flow_cols}
    depth_all = (st["bid_qty"].to_numpy(dtype="float64")
                 + st["ask_qty"].to_numpy(dtype="float64"))
    spread = st["spread"].to_numpy(dtype="float64")

    w_max = max(C.PRE_WINDOWS_S)
    h_max = max(C.HORIZONS_S)
    t0 = np.arange(w_max, n - h_max, dtype="int64")

    def wsum(k, w):
        return cum[k][t0 + 1] - cum[k][t0 + 1 - w]     # Sekunden [t0-w+1, t0]

    out = {"date": date, "t0_sec": t0, "mid_t0": mid[t0]}
    out["depth"] = depth_all[t0]
    out["spread_bps"] = spread[t0] / np.maximum(mid[t0], EPS) * 1e4

    for w in C.PRE_WINDOWS_S:
        sb, sa = wsum("stack_bid", w), wsum("stack_ask", w)
        pb, pa = wsum("pull_bid", w), wsum("pull_ask", w)
        d_end, d_start = depth_all[t0], depth_all[t0 - w + 1]
        out[f"rv_pre_{w}"] = rv(t0 - w + 1, t0)
        out[f"netout_{w}"] = (pb + pa) - (sb + sa)
        out[f"depth_ratio_{w}"] = d_end / np.maximum(d_start, EPS)
        out[f"churn_{w}"] = (pb + pa + sb + sa) / np.maximum(d_end, EPS)
        out[f"vol_{w}"] = wsum("trade_vol", w)
        out[f"ntrades_{w}"] = wsum("n_trades", w)
        # Netto-Abfluss relativ zur Tiefe (skalenfrei, fuer die Regression).
        out[f"netout_norm_{w}"] = out[f"netout_{w}"] / np.maximum(d_end, EPS)

    # Ziele NACH t0: Sekunden [t0+1, t0+h].
    mid_s = pd.Series(mid)
    for h in C.HORIZONS_S:
        out[f"rv_{h}"] = rv(t0 + 1, t0 + h)
        fmax = mid_s.rolling(h).max().shift(-h).to_numpy()[t0]
        fmin = mid_s.rolling(h).min().shift(-h).to_numpy()[t0]
        out[f"range_{h}"] = (fmax - fmin) / np.maximum(mid[t0], EPS) * 1e4

    df = pd.DataFrame(out)
    df = df[np.isfinite(df["mid_t0"]) & (df["mid_t0"] > 0)]
    for h in C.HORIZONS_S:
        df = df[np.isfinite(df[f"rv_{h}"]) & np.isfinite(df[f"range_{h}"])]
    return df.reset_index(drop=True)


# ===========================================================================
#  Regression (ohne externe Abhaengigkeit)
# ===========================================================================
def _design(df: pd.DataFrame, feats: list[str], w: int) -> np.ndarray:
    """Merkmalsmatrix bauen; fensterabhaengige Namen bekommen das Suffix _w."""
    cols = []
    for f in feats:
        if f == "log_rv_pre":
            x = np.log1p(df[f"rv_pre_{w}"].to_numpy())
        elif f == "log_vol":
            x = np.log1p(df[f"vol_{w}"].to_numpy())
        elif f == "log_ntrades":
            x = np.log1p(df[f"ntrades_{w}"].to_numpy())
        elif f == "netout_norm":
            x = df[f"netout_norm_{w}"].to_numpy()
        elif f == "depth_ratio":
            x = df[f"depth_ratio_{w}"].to_numpy()
        elif f == "churn":
            x = df[f"churn_{w}"].to_numpy()
        elif f == "log_depth":
            x = np.log1p(df["depth"].to_numpy())
        elif f == "spread_bps":
            x = df["spread_bps"].to_numpy()
        else:
            raise KeyError(f)
        cols.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    return np.column_stack(cols)


def _fit_predict(Xtr, ytr, Xte):
    """OLS mit Standardisierung (Statistiken NUR aus dem Trainingsteil)."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    Ztr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    Zte = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    beta, *_ = np.linalg.lstsq(Ztr, ytr, rcond=None)
    return Zte @ beta


def ladder(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Die Treppe mit Leave-one-day-out-Kreuzvalidierung.

    Fuer jede Kombination (Fenster x Horizont x Zielgroesse x Stufe):
      - nur nicht-ueberlappende Bloecke (Schrittweite = Horizont)
      - trainieren auf allen Tagen ausser einem, testen auf dem ausgelassenen
      - R2 out-of-sample (Referenz: Mittelwert des Trainingsteils) und MAE in bps
    """
    rows = []
    dates = list(samples.keys())
    for w in C.PRE_WINDOWS_S:
        for h in C.HORIZONS_S:
            # Nicht-ueberlappende Bloecke -> unabhaengige Beobachtungen.
            blocks = {d: s[s["t0_sec"] % h == 0] for d, s in samples.items()}
            for tgt in TARGETS:
                ycol = f"{tgt}_{h}"
                for stage_name, feats in STAGES.items():
                    preds, actuals, n_train_total = [], [], 0
                    for held in dates:
                        tr = pd.concat([blocks[d] for d in dates if d != held],
                                       ignore_index=True) if len(dates) > 1 else blocks[held]
                        te = blocks[held]
                        if len(tr) < 50 or len(te) < 10:
                            continue
                        ytr = np.log1p(tr[ycol].to_numpy())
                        yte = te[ycol].to_numpy()
                        Xtr, Xte = _design(tr, feats, w), _design(te, feats, w)
                        pred_log = _fit_predict(Xtr, ytr, Xte)
                        preds.append(np.expm1(pred_log))
                        actuals.append(yte)
                        n_train_total += len(tr)
                    if not preds:
                        continue
                    pred = np.concatenate(preds)
                    act = np.concatenate(actuals)
                    pred = np.clip(np.nan_to_num(pred, nan=0.0), 0.0, None)
                    ss_res = np.sum((act - pred) ** 2)
                    ss_tot = np.sum((act - act.mean()) ** 2)
                    rows.append({
                        "pre_s": w, "horizon_s": h, "ziel": tgt, "stufe": stage_name,
                        "n_eff": len(act),
                        "belastbar": len(act) >= 30,
                        "r2_oos": 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan,
                        "mae_bps": float(np.mean(np.abs(act - pred))),
                        "ziel_mittel_bps": float(act.mean()),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Verbesserung je Stufe gegenueber der vorigen Stufe explizit ausweisen.
    order = list(STAGES.keys())
    df["stufe_idx"] = df["stufe"].map({s: i for i, s in enumerate(order)})
    df = df.sort_values(["pre_s", "horizon_s", "ziel", "stufe_idx"]).reset_index(drop=True)
    key = ["pre_s", "horizon_s", "ziel"]
    df["d_r2_vs_vorstufe"] = df.groupby(key)["r2_oos"].diff()
    df["d_mae_vs_vorstufe"] = df.groupby(key)["mae_bps"].diff()
    return df


# ===========================================================================
#  Terzile nach Netto-Abfluss
# ===========================================================================
def terciles(samples: dict[str, pd.DataFrame], per_day: bool = False) -> pd.DataFrame:
    """
    Zeitpunkte nach Netto-Abfluss in Terzile teilen und die durchschnittliche
    Folge-Volatilitaet je Terzil zeigen.

    Die Terzilgrenzen werden INNERHALB jedes Tages bestimmt. Grund: die absoluten
    Groessenordnungen unterscheiden sich stark zwischen Wochentagen und Wochenende;
    gepoolte Grenzen wuerden vor allem Tage trennen, nicht Marktzustaende.
    """
    rows = []
    # qcut vergibt das ERSTE Label an die NIEDRIGSTEN Werte. netout positiv =
    # Abfluss, also: niedrigster netout = aufbauend, hoechster = abfliessend.
    labels = ["stark aufbauend", "neutral", "stark abfliessend"]
    for w in C.PRE_WINDOWS_S:
        for h in C.HORIZONS_S:
            per_day_parts = []
            for d, s in samples.items():
                b = s[s["t0_sec"] % h == 0].copy()
                if len(b) < 30:
                    continue
                try:
                    b["terzil"] = pd.qcut(b[f"netout_{w}"], 3, labels=labels)
                except ValueError:
                    continue
                per_day_parts.append(b)
                if per_day:
                    for lab in labels:
                        g = b[b["terzil"] == lab]
                        rows.append({"scope": d, "pre_s": w, "horizon_s": h, "terzil": lab,
                                     "n_eff": len(g),
                                     "belastbar": len(g) >= 30,
                                     "rv_mittel": float(g[f"rv_{h}"].mean()) if len(g) else np.nan,
                                     "range_mittel": float(g[f"range_{h}"].mean()) if len(g) else np.nan,
                                     "netout_mittel": float(g[f"netout_{w}"].mean()) if len(g) else np.nan})
            if per_day_parts and not per_day:
                allb = pd.concat(per_day_parts, ignore_index=True)
                for lab in labels:
                    g = allb[allb["terzil"] == lab]
                    rows.append({"scope": "GEPOOLT", "pre_s": w, "horizon_s": h, "terzil": lab,
                                 "n_eff": len(g),
                                 "belastbar": len(g) >= 30,
                                 "rv_mittel": float(g[f"rv_{h}"].mean()) if len(g) else np.nan,
                                 "range_mittel": float(g[f"range_{h}"].mean()) if len(g) else np.nan,
                                 "netout_mittel": float(g[f"netout_{w}"].mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def n_variants(n_days: int) -> dict:
    n_cells = len(C.PRE_WINDOWS_S) * len(C.HORIZONS_S) * len(TARGETS)
    return {
        "fenster": len(C.PRE_WINDOWS_S),
        "horizonte": len(C.HORIZONS_S),
        "zielgroessen": len(TARGETS),
        "stufen": len(STAGES),
        "zellen_gesamt": n_cells,
        "modellanpassungen": n_cells * len(STAGES) * n_days,   # LODO
        "terzil_zellen": len(C.PRE_WINDOWS_S) * len(C.HORIZONS_S) * 3,
    }
