"""
Regressionstest: Die Volatilitaets-Treppe muss bei pathologischen Tiefen
(winzige Restmengen, Nullen) endliche und plausible Ergebnisse liefern.

Hintergrund: In frueheren Laeufen entstanden aus Quotienten wie
depth_ratio = d_end / d_start bei d_start ~ 1e-9 Werte um 1e12. Die sind
endlich, werden also von nan_to_num nicht abgefangen, verzerren aber die
Regression massiv (R2 brach auf -227 ein). Winsorisieren allein genuegt nicht:
liegt mehr als ein halbes Prozent der Werte im Extrem, ist schon das
99.5-Perzentil astronomisch. Deshalb harte Grenzen bei der Konstruktion.

Geprueft wird das ERGEBNIS, nicht das Ausbleiben von Warnungen: manche
BLAS-Implementierungen (u. a. Apple Accelerate auf macOS) melden bei matmul
Fliesskomma-Flags aus ungenutzten SIMD-Lanes, auch wenn die Rechnung harmlos
ist. Diese Warnungen sind eine Plattform-Eigenheit und kein Auswertungsfehler.

Ausfuehren:  python -m pytest tests/ -q
"""



import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import vola


def _pathological_seconds(n=N if (N := 24 * 3600) else 0, seed=0):
    """Sekundentabelle mit absichtlich kaputten Tiefen."""
    rng = np.random.default_rng(seed)
    mid = 77.0 + np.cumsum(rng.normal(0, 0.002, n))
    bid = rng.uniform(100, 500, n)
    ask = rng.uniform(100, 500, n)
    # 3 % der Sekunden mit winzigen bzw. null Tiefen - genau der Fall, der
    # frueher Werte um 1e12 erzeugt hat (mehr als 0.5 %, also unwinsorisierbar).
    bad = rng.random(n) < 0.03
    bid[bad] = rng.choice([0.0, 1e-9, 1e-12], size=bad.sum())
    ask[bad] = rng.choice([0.0, 1e-9, 1e-12], size=bad.sum())
    return pd.DataFrame({
        "mid": mid, "spread": np.full(n, 0.01), "bid_qty": bid, "ask_qty": ask,
        "stack_bid": rng.uniform(0, 50, n), "stack_ask": rng.uniform(0, 50, n),
        "pull_bid": rng.uniform(0, 50, n), "pull_ask": rng.uniform(0, 50, n),
        "buy_vol": rng.uniform(0, 20, n), "sell_vol": rng.uniform(0, 20, n),
        "trade_vol": rng.uniform(0, 40, n), "n_trades": rng.integers(0, 10, n).astype(float),
        "n_dec": rng.integers(0, 5, n).astype(float), "n_amb": np.zeros(n),
    })


def test_sample_features_sind_begrenzt():
    """Quotienten-Merkmale muessen endlich und innerhalb der harten Grenzen sein."""
    st = _pathological_seconds()
    s = vola.build_vola_sample(st, "2026-07-15")
    assert len(s) > 1000, "Sample sollte nicht leer sein"
    for w in C.PRE_WINDOWS_S:
        for col, limit in [(f"depth_ratio_{w}", vola.DEPTH_RATIO_CLIP),
                           (f"churn_{w}", vola.CHURN_CLIP),
                           (f"netout_norm_{w}", vola.NETOUT_NORM_CLIP)]:
            x = s[col].to_numpy()
            assert np.isfinite(x).all(), f"{col} enthaelt nicht-endliche Werte"
            assert np.abs(x).max() <= limit * 1.0001, f"{col} ueberschreitet die Grenze"


def test_treppe_liefert_plausible_ergebnisse():
    """
    Der komplette Treppen-Lauf muss endliche, plausible Werte liefern.

    Bewusst wird NICHT auf das Ausbleiben von RuntimeWarnings geprueft: manche
    BLAS-Implementierungen (u. a. Apple Accelerate) melden bei matmul
    Fliesskomma-Flags aus ungenutzten SIMD-Lanes, obwohl die Rechnung harmlos
    ist. Das ist eine Plattform-Eigenheit, kein Fehler in dieser Auswertung.
    Geprueft wird deshalb das ERGEBNIS - dort schlaegt ein echtes numerisches
    Problem zuverlaessig durch.
    """
    samples = {d: vola.build_vola_sample(_pathological_seconds(seed=i), d)
               for i, d in enumerate(["2026-07-15", "2026-07-16", "2026-07-17"])}
    lad = vola.ladder(samples)
    assert not lad.empty
    assert lad["r2_oos"].notna().all(), "R2 darf nicht NaN sein"
    assert lad["mae_bps"].notna().all(), "MAE darf nicht NaN sein"
    assert np.isfinite(lad["mae_bps"]).all(), "MAE muss endlich sein"
    # Plausibilitaet: R2 muss im sinnvollen Bereich liegen, nicht bei -227.
    assert lad["r2_oos"].min() > -5.0, "R2 unplausibel negativ - Hinweis auf Ueberlauf"


def test_fit_predict_faengt_echte_ueberlaeufe():
    """Ein echtes numerisches Problem muss einen Fehler ausloesen, nicht still passieren."""
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(0.05, 0.4, (500, 1))
    ytr = rng.uniform(0.1, 0.3, 500)
    # harmlose Eingaben -> endliches Ergebnis, kein Fehler
    pred = vola._fit_predict(Xtr, ytr, rng.uniform(0.05, 0.4, (100, 1)))
    assert np.isfinite(pred).all()
    # Unendliche Merkmale werden auf die Trainingsgrenzen geklippt und sind
    # damit unschaedlich - das Ergebnis bleibt endlich.
    assert np.isfinite(vola._fit_predict(Xtr, ytr, np.full((10, 1), np.inf))).all()
    # Ein korruptes Ziel dagegen macht die Koeffizienten unbrauchbar: das MUSS
    # auffallen und darf nicht still als Ergebnis durchgehen.
    y_bad = ytr.copy()
    y_bad[0] = np.nan
    with pytest.raises(FloatingPointError):
        vola._fit_predict(Xtr, y_bad, Xtr[:10])


def test_kein_lookahead_in_vorfenster():
    """rv_pre darf sich nicht aendern, wenn man die Zukunft nach t0 veraendert."""
    st = _pathological_seconds(seed=7)
    s1 = vola.build_vola_sample(st, "2026-07-15")
    st2 = st.copy()
    cut = 60000
    st2.loc[cut:, "mid"] = st2.loc[cut:, "mid"] * 1.05   # Zukunft veraendern
    s2 = vola.build_vola_sample(st2, "2026-07-15")
    m = s1["t0_sec"] < cut - max(C.HORIZONS_S) - 5
    a = s1.loc[m].set_index("t0_sec")[f"rv_pre_{C.PRE_WINDOWS_S[-1]}"]
    b = s2.set_index("t0_sec").reindex(a.index)[f"rv_pre_{C.PRE_WINDOWS_S[-1]}"]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), \
        "Vorfenster-Merkmal haengt von der Zukunft ab -> Look-ahead!"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
