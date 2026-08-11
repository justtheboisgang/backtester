"""
Zentrale Konfiguration der Studie.

Hier stehen ALLE methodischen Entscheidungen an einem Ort, damit sie
ueberpruefbar und nicht im Code verstreut sind. Aendere Parameter nur hier.

Forschungsfrage:
  Zeigen sich im Orderbuch von SOL-Perps an VAH/VAL der VORHERIGEN Session
  systematische Pull/Stack-Ungleichgewichte, BEVOR der Preis sich bewegt -
  und ist der Move nach Kosten handelbar?
"""

from __future__ import annotations
from pathlib import Path
import cryptohftdata as chd

# ---------------------------------------------------------------------------
# 1) Markt & Zeitraum
# ---------------------------------------------------------------------------
SYMBOL = "SOLUSDT"
EXCHANGE = chd.exchanges.BINANCE_FUTURES        # "binance_futures"

# Wir fangen bewusst mit EINEM Tag an. Erst wenn die Pipeline sauber laeuft
# und du das Ergebnis gesehen hast, erweitern wir auf mehrere Tage.
START_DATE = "2026-07-15"
END_DATE = "2026-07-15"                          # inklusiv, ganzer Tag (UTC)

# ---------------------------------------------------------------------------
# 2) Sessions (UTC). Grenzen als [start_h, end_h) in Stunden.
#    Reihenfolge = zeitliche Reihenfolge am Tag.
# ---------------------------------------------------------------------------
SESSIONS = [
    ("asia",   0,  8),
    ("london", 8,  13),
    ("us",     13, 24),
]

# Value Area = 70 % des gehandelten Volumens rund um den POC.
VALUE_AREA_PCT = 0.70
# Preis-Bin fuer das Volumenprofil (in USDT). SOL ~ 150 USD -> 0.05 ist stabil.
VA_PRICE_BIN = 0.05

# ---------------------------------------------------------------------------
# 3) Event-Definition
# ---------------------------------------------------------------------------
# Ein Event = der Preis erreicht ein Referenzlevel (VAH/VAL) der VORIGEN Session.
# Band um das Level (in USDT): so nah muss der Preis kommen, damit es als
# "Level erreicht" zaehlt, und in diesem Band messen wir die Orderbuch-Aktivitaet.
LEVEL_BAND = 0.05
# Abklingzeit (Sekunden): nach einem Event am selben Level erst wieder ein neues
# Event zulassen, wenn der Preis das Band verlassen hatte. Verhindert, dass
# Mikro-Zappeln am Level Dutzende Events erzeugt.
EVENT_COOLDOWN_S = 120

# Vor-Fenster (Sekunden): Zeitraum VOR t0, in dem wir Pull/Stack messen.
PRE_WINDOWS_S = [5, 15, 30]
# Nach-Fenster (Sekunden): Zeitraum NACH t0, in dem wir die Preisbewegung messen.
POST_HORIZONS_S = [30, 60, 300]

# ---------------------------------------------------------------------------
# 4) Pull vs. Fill (Anforderung 1)
# ---------------------------------------------------------------------------
# Eine schrumpfende Menge auf einem Level ist Storno (Pull) ODER Ausfuehrung (Fill).
# Wir gleichen mit dem Trades-Feed ab:
#   Trade zur ~selben Zeit auf ~demselben Preis  -> Fill (Ausfuehrung)
#   sonst                                        -> Pull (Storno)
TRADE_MATCH_TOL_MS = 100      # zeitliche Toleranz fuer das Matching
TRADE_MATCH_PRICE_DP = 3      # Preis auf so viele Nachkommastellen runden beim Matchen
# Wenn eine Mengenreduktion nur teilweise durch Trades erklaerbar ist und der
# Rest kleiner als dieser Anteil der Reduktion ist, gilt sie als "eindeutig Fill".
FILL_TOLERANCE_FRAC = 0.10

# ---------------------------------------------------------------------------
# 5) Kontrollgruppe (Anforderung 4)
# ---------------------------------------------------------------------------
N_CONTROL = 1000              # zufaellige Zeitpunkte OHNE Levelbezug
CONTROL_MIN_DIST = 0.25       # Kontrollpunkt muss > so viele USDT von jedem Level weg sein
RANDOM_SEED = 42              # reproduzierbar

# ---------------------------------------------------------------------------
# 6) Kosten (Anforderung 6) - alles in Basispunkten (bps), 1 bps = 0.01 %
# ---------------------------------------------------------------------------
# Binance USDT-M Futures Standard-Taker = 0.05 % = 5 bps pro Seite.
TAKER_FEE_BPS = 5.0
ROUND_TRIP_FEE_BPS = 2 * TAKER_FEE_BPS   # rein + raus
# Der durchschnittliche Spread wird aus den Daten geschaetzt (nicht angenommen).

# ---------------------------------------------------------------------------
# 6b) DOM-Studie (zweite Auswertung): DOM-Zustand -> Vorwaertsbewegung
# ---------------------------------------------------------------------------
# Zeithorizonte der Vorwaertsbewegung (Sekunden).
HORIZONS_S = [30, 60, 120]
# Kostenschwellen fuer "grosser Move" (Anforderung):
#   Taker = 10 bps Gebuehr + Spread  ~ 12 bps
#   Maker =  4 bps Gebuehr + halber Spread ~ 6 bps
THRESH_TAKER_BPS = 12.0
THRESH_MAKER_BPS = 6.0
# Band um den Mid (USDT), in dem Orderbuch-Aktivitaet dem "Buch" zugerechnet wird.
DOM_BOOK_BAND = 0.05
# Naehe zu VAH/VAL der Vor-Session fuer die Teilmengen-Markierung.
DOM_NEAR_LEVEL_BAND = 0.05
# Standard-Anzahl Tage fuer die Studie (Tag fuer Tag verarbeitet).
STUDY_DAYS = 5

# ---------------------------------------------------------------------------
# 7) Pfade  (data/ ist in .gitignore -> bleibt lokal)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # Rohdaten wie vom Anbieter geladen
INTERIM_DIR = DATA_DIR / "interim"  # Zwischenstaende der Schritte
OUTPUT_DIR = DATA_DIR / "output"    # Endergebnisse / Report

for _d in (RAW_DIR, INTERIM_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def tag() -> str:
    """Kurzkennung fuer Dateinamen, z. B. 'SOLUSDT_2026-07-15'."""
    if START_DATE == END_DATE:
        return f"{SYMBOL}_{START_DATE}"
    return f"{SYMBOL}_{START_DATE}_bis_{END_DATE}"
