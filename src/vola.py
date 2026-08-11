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

# Harte, sachlich begruendete Grenzen fuer die Quotienten-Merkmale.
# Ohne sie entstehen aus winzigen Restmengen im Nenner Werte um 1e12 und groesser.
# Winsorisieren allein reicht dagegen NICHT: liegt mehr als ein halbes Prozent der
# Werte im Extrem, ist schon das 99.5-Perzentil astronomisch. Und ein einzelner
# Wert von 1e200 laesst bereits die Standardabweichung ueberlaufen.
DEPTH_FLOOR = 1e-6        # darunter gilt die Tiefe als nicht auswertbar
DEPTH_RATIO_CLIP = 100.0  # Tiefenaenderung um mehr als Faktor 100 in <=30 s ist ein Artefakt
CHURN_CLIP = 1000.0       # Umschichtung > 1000x der Tiefe ist ein Artefakt
NETOUT_NORM_CLIP = 1000.0

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

    # Quotienten nur bilden, wo die Tiefe ueber dem Boden liegt, und das Ergebnis
    # hart begrenzen (siehe Kommentar bei den Konstanten oben).
    valid_depth = depth_all > DEPTH_FLOOR

    for w in C.PRE_WINDOWS_S:
        sb, sa = wsum("stack_bid", w), wsum("stack_ask", w)
        pb, pa = wsum("pull_bid", w), wsum("pull_ask", w)
        d_end, d_start = depth_all[t0], depth_all[t0 - w + 1]
        ok = valid_depth[t0] & valid_depth[t0 - w + 1]
        safe_end = np.where(ok, d_end, 1.0)
        safe_start = np.where(ok, d_start, 1.0)
        out[f"rv_pre_{w}"] = rv(t0 - w + 1, t0)
        out[f"netout_{w}"] = (pb + pa) - (sb + sa)
        out[f"depth_ratio_{w}"] = np.where(
            ok, np.clip(d_end / safe_start, 1.0 / DEPTH_RATIO_CLIP, DEPTH_RATIO_CLIP), np.nan)
        out[f"churn_{w}"] = np.where(
            ok, np.clip((pb + pa + sb + sa) / safe_end, 0.0, CHURN_CLIP), np.nan)
        out[f"vol_{w}"] = wsum("trade_vol", w)
        out[f"ntrades_{w}"] = wsum("n_trades", w)
        # Netto-Abfluss relativ zur Tiefe (skalenfrei, fuer die Regression).
        out[f"netout_norm_{w}"] = np.where(
            ok, np.clip((out[f"netout_{w}"]) / safe_end,
                        -NETOUT_NORM_CLIP, NETOUT_NORM_CLIP), np.nan)

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
    # Zeitpunkte ohne gueltige Tiefe verwerfen, statt sie mit Ersatzwerten
    # weiterzuschleppen - sie wuerden die Regression dominieren.
    feat_cols = [f"{p}_{w}" for w in C.PRE_WINDOWS_S
                 for p in ("depth_ratio", "churn", "netout_norm")]
    df = df[np.isfinite(df[feat_cols]).all(axis=1) & np.isfinite(df["depth"]) & (df["depth"] > 0)]
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
    """
    OLS mit Winsorisierung und Standardisierung.
    Alle Statistiken (Kappungsgrenzen, Mittelwert, Streuung) stammen
    AUSSCHLIESSLICH aus dem Trainingsteil - sonst waere es Look-ahead.
    """
    # Extremwerte kappen: einzelne Ausreisser wuerden die Koeffizienten sonst
    # dominieren und die Vorhersage auf dem Testtag unbrauchbar machen.
    lo = np.percentile(Xtr, 0.5, axis=0)
    hi = np.percentile(Xtr, 99.5, axis=0)
    Xtr = np.clip(Xtr, lo, hi)
    Xte = np.clip(Xte, lo, hi)

    # Robuste Skalierung ueber Median und Interquartilsabstand: Mittelwert und
    # Standardabweichung koennen bei sehr grossen Werten selbst ueberlaufen
    # (x^2 laeuft schon ab ~1e154 ueber), der Median nicht.
    med = np.median(Xtr, axis=0)
    q75, q25 = np.percentile(Xtr, 75, axis=0), np.percentile(Xtr, 25, axis=0)
    scale = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)

    def z(X):
        Z = (X - med) / scale
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return np.column_stack([np.ones(len(Z)), Z])

    Ztr, Zte = z(Xtr), z(Xte)
    beta, *_ = np.linalg.lstsq(Ztr, ytr, rcond=1e-10)

    # Manche BLAS-Implementierungen (u. a. Apple Accelerate auf macOS) setzen bei
    # matmul Fliesskomma-Flags auf ungenutzten SIMD-Lanes und loesen dadurch
    # RuntimeWarnings aus, obwohl die Rechnung voellig harmlos ist. Deshalb hier
    # die Flags stummschalten - aber NICHT blind: direkt danach wird geprueft,
    # ob das Ergebnis tatsaechlich endlich ist. Echte Probleme fallen so weiter auf.
    with np.errstate(all="ignore"):
        pred = Zte @ beta
    if not np.isfinite(pred).all():
        n_bad = int((~np.isfinite(pred)).sum())
        raise FloatingPointError(
            f"Vorhersage enthaelt {n_bad} nicht-endliche Werte - "
            f"das ist ein echtes numerisches Problem, keine Plattform-Warnung."
        )
    return pred


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


def netout_diagnosis(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prueft, ob 'netout' ueberhaupt eigenstaendige Information traegt.

    Hintergrund: netout = pull - stack zaehlt bei den Abgaengen NUR Stornos,
    denn Ausfuehrungen wurden herausklassifiziert. Ueber ein Fenster gilt
        Tiefenaenderung = stack - pull - fills.
    Bleibt die Tiefe ungefaehr konstant, folgt stack - pull ~ fills, also
        netout ~ -Handelsvolumen.
    Ist die Korrelation zwischen netout und Volumen stark negativ, misst netout
    im Kern nur das Handelsvolumen - und ist damit kein eigenstaendiges
    Orderbuch-Signal.
    """
    rows = []
    pooled = pd.concat(samples.values(), ignore_index=True)
    for w in C.PRE_WINDOWS_S:
        x = pooled[f"netout_{w}"].to_numpy(dtype="float64")
        v = pooled[f"vol_{w}"].to_numpy(dtype="float64")
        ok = np.isfinite(x) & np.isfinite(v)
        corr = float(np.corrcoef(x[ok], v[ok])[0, 1]) if ok.sum() > 2 else np.nan
        # Anteil der Zeitpunkte mit netout < 0 (mehr Aufbau als Storno).
        rows.append({"pre_s": w, "n": int(ok.sum()),
                     "corr_netout_volumen": corr,
                     "anteil_netout_negativ": float((x[ok] < 0).mean()),
                     "netout_median": float(np.median(x[ok])),
                     "volumen_median": float(np.median(v[ok]))})
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
