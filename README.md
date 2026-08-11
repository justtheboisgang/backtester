# SOL-Perps Orderbuch-Studie: Pull/Stack an VAH/VAL

**Forschungsfrage:** Zeigen sich im Orderbuch von SOL-Perps (Binance Futures,
`SOLUSDT`) an **VAH/VAL der vorherigen Session** systematische
**Pull/Stack-Ungleichgewichte, bevor** der Preis sich bewegt — und ist der
anschließende Move **nach Kosten** handelbar?

Diese Pipeline beantwortet das in **kleinen, einzeln überprüfbaren Schritten**.
Nach jedem Schritt gibt es eine Diagnose (Zeilenzahl, Speicher, Zeitraum) und
ein Parquet-Zwischenergebnis, damit nichts doppelt geladen wird.

> **Ehrliche Vorab-Warnung:** Die mitgelieferten Beispiel-Zahlen stammen aus
> **synthetischen Testdaten** (Zufall, `--synth`). Sie beweisen nur, dass die
> *Logik* sauber läuft — **kein** Handelssignal. Echte Ergebnisse bekommst du
> erst mit echten Daten (siehe „Echte Daten laden").

---

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. API-Key einrichten (nur für echte Daten)

Der Key kommt in eine `.env` und **niemals** in den Code (die `.env` ist per
`.gitignore` ausgeschlossen).

```bash
cp .env.example .env
# .env öffnen und CRYPTOHFTDATA_API_KEY=... eintragen
```

Den Key gibt es bei <https://www.cryptohftdata.com/>.

## 3. Pipeline testen — ohne Key, ohne Netzwerk (empfohlen zum Kennenlernen)

```bash
python run_all.py --synth
```

Das erzeugt synthetische Daten und lässt **alle 6 Schritte** durchlaufen. So
siehst du Format und Diagnose jedes Schritts, bevor echte Daten fließen.

## 4. Echte Daten laden

```bash
python run_all.py            # lädt echte Daten via cryptohftdata (Key aus .env)
```

oder Schritt für Schritt (jeder Schritt liest den Parquet-Cache des vorigen):

```bash
python steps/step01_load.py            # Rohdaten (Orderbuch + Trades)
python steps/step02_value_area.py      # Value Area je Session + Referenz-Zuordnung
python steps/step03_events.py          # Events: Preis erreicht Referenzlevel
python steps/step04_top_of_book.py     # Top-of-Book -> Spread (für Kosten)
python steps/step05_measure.py         # Imbalance (vorher) + Move (nachher), Events & Kontrolle
python steps/step06_report.py          # Report: Fallzahlen + Kostenvergleich
```

`--force` an jedem Skript ignoriert den Cache und rechnet neu.

---

## ⚠️ Wichtig: Netzwerk-Sperre in der Cloud-Session

Diese Pipeline wurde in einer Claude-Cloud-Umgebung gebaut. Dort ist der Host
`api.cryptohftdata.com` durch die **Egress-Policy blockiert** (403). Deshalb
konnte der **echte Download hier nicht laufen** — er muss **auf deiner Maschine**
laufen, wo die Domain erreichbar ist und dein Key in der `.env` liegt.

Verifiziert wurde die gesamte Methodik hier **offline** über den synthetischen
Generator (`--synth`).

---

## ⚠️ Wichtig: Schema-Abweichung prüfen (Schritt 1)

Der Prompt beschreibt das Orderbuch als Event-Stream (`event_time`,
`event_type` „snapshot"/„update", `side`, `price`, `quantity`). Das installierte
Paket (`cryptohftdata==0.4.0`) nennt als typische Spalten dagegen
`timestamp, side, level, price, size` und weist selbst darauf hin, dass Spalten
**je Exchange abweichen** können.

Deshalb ist der Loader (`src/chd_io.py`) **schema-tolerant**: Er erkennt die
echten Spalten zur Laufzeit, übersetzt sie auf ein festes kanonisches Schema und
**druckt beim ersten echten Download die ROH-Spalten aus**. Bitte beim ersten
echten Lauf die Zeile

```
[normalize_orderbook] ROH-Spalten: [...]
```

anschauen. Passt eine Spalte nicht in die Kandidatenlisten (`_pick(...)` in
`chd_io.py`), meldet der Loader das mit einer klaren Fehlermeldung — dann dort
den echten Spaltennamen ergänzen. **`quantity` muss die absolute Restmenge auf
dem Level sein** (0 = Level gelöscht); nur dann stimmt die Pull/Stack-Logik.

---

## Kanonisches Schema (worauf die Analyse rechnet)

| Datensatz | Spalten |
|---|---|
| Orderbuch (`ob`) | `event_time`, `event_type` (`snapshot`/`update`), `side` (`bid`/`ask`), `price`, `quantity` (absolute Restmenge; 0 = gelöscht) |
| Trades (`tr`) | `event_time`, `side` (`buy`/`sell` = Aggressor), `price`, `quantity` |

---

## Wie die 7 methodischen Anforderungen umgesetzt sind

1. **Pull von Trade trennen** — `src/book.py: classify_arrays`. Jede
   Mengenreduktion auf einem Level wird gegen den Trades-Feed abgeglichen:
   Trade am selben Preis (`±0,001` gerundet) innerhalb `±100 ms`
   → **Fill**, sonst → **Pull**; teilweise gedeckt → **ambiguous**. Der
   **Anteil unklarer Fälle** wird in Schritt 5 explizit ausgewiesen.
2. **Kein Look-ahead** — `pipeline.build_reference_map`. Die Value Area einer
   Session (Asien 00–08, London 08–13, US 13–24 UTC) dient nur als Referenz für
   die **folgende** Session. Am ersten Tag hat Asien keinen Vorgänger und wird
   weggelassen (bei mehreren Tagen nutzt Asien die US-Session des Vortags).
3. **Event-Definition & mehrere Fenster** — `pipeline.detect_events` +
   `Measurer`. Event = Preis erreicht ein Referenzlevel. Vor-Fenster
   `5/15/30 s`, Nach-Fenster `30/60/300 s` (in `config.py`). Die
   Preisbewegung wird auf dem **Mid-Preis** aus dem rekonstruierten Orderbuch
   gemessen (Schritt 4) — robust gegen fehlerhafte Trade-Prints (Ausreißer),
   die den Move-Mittelwert sonst verzerren.
4. **Kontrollgruppe (Pflicht)** — `pipeline.build_controls`. Dieselbe Messung an
   zufälligen Zeitpunkten **ohne** Levelbezug (mind. `0,25` USDT von jedem Level
   entfernt).
5. **Fallzahlen** — `pipeline.build_report`. Spalte `belastbar` = `False`, sobald
   eine Gruppe unter 30 Fällen liegt.
6. **Kosten** — `pipeline.build_report`. Mittlerer `|Move|` steht immer neben den
   Kosten = `2 × Taker (5 bps) + mittlerer Spread` (aus den Daten geschätzt).
   Spalte `handelbar` = `False`, wenn `|Move| ≤ Kosten`.
7. **Kein Nachtunen** — Alle Parameter stehen fest in `config.py`. Der Report
   weist `n_tests_total` aus (Anzahl getesteter Fenster-Kombinationen), damit
   Multiple Testing sichtbar bleibt. Ergebnisse werden berichtet, wie sie
   herauskommen.

---

## Projektstruktur

```
config.py-Parameter  ->  src/config.py     (alle Entscheidungen an einem Ort)
Laden/Normalisieren  ->  src/chd_io.py     (schema-tolerant + Diagnose + Cache)
Orderbuch-Mechanik   ->  src/book.py       (Top-of-Book, Pull/Fill-Trennung)
Analyse-Kern         ->  src/pipeline.py   (VA, Events, Messung, Kontrolle, Report)
synthetische Daten   ->  src/synth.py      (nur zum Offline-Testen)
Orchestrierung       ->  src/steps.py      (eine Funktion je Schritt)
Schritt-Skripte      ->  steps/step01..06  (einzeln ausführbar)
alles am Stück       ->  run_all.py
Zwischenstände       ->  data/  (raw/ interim/ output/, per .gitignore lokal)
```

## Report-Spalten (Schritt 6)

| Spalte | Bedeutung |
|---|---|
| `pre_s`, `post_s` | Vor-/Nach-Fenster in Sekunden |
| `n_events`, `n_control` | Fallzahlen je Gruppe |
| `belastbar` | `True`, wenn beide Gruppen ≥ 30 Fälle |
| `imb_event_mean`, `imb_control_mean` | mittlere Pull/Stack-Imbalance (−1…+1; + = Bids stärker/Asks weg) |
| `corr_imb_move_event` | Korrelation Imbalance ↔ signierter Move (Events) |
| `abs_move_event_bps`, `abs_move_control_bps` | mittlerer \|Move\| in bps |
| `spread_bps`, `cost_bps` | geschätzter Spread bzw. Gesamtkosten |
| `net_event_bps`, `handelbar` | \|Move\| minus Kosten; handelbar nur wenn > 0 |
| `n_tests_total` | Anzahl getesteter Fenster-Kombinationen |

---

---

# Zweite Studie: DOM-Zustand → Vorwärtsbewegung

**Forschungsfrage:** Zeigt der DOM-Zustand (Pull/Stack je Seite, Imbalance,
Spread, Tiefe) **vor** einer Bewegung ein wiederkehrendes Muster — (a) dass
gleich eine **größere** Bewegung kommt, und (b) in welche **Richtung**?
Horizonte **30 / 60 / 120 s**.

```bash
python run_dom_study.py --days 5                    # echte Daten, Tag für Tag
python run_dom_study.py --days 5 --start 2026-07-15
python run_dom_study.py --days 3 --synth            # offline testen
python run_dom_study.py --analyze-only              # nur Analyse aus Cache
```

## Vorzeichen-Konvention (wichtig)

```
imbalance = ((stack_bid + pull_ask) − (stack_ask + pull_bid)) / Summe aller vier
```

| Vorzeichen | Bedeutung |
|---|---|
| **positiv** | **Aufwärtsdruck**: Bids werden **aufgebaut** und/oder Asks **abgezogen** |
| **negativ** | **Abwärtsdruck**: Asks werden **aufgebaut** und/oder Bids **abgezogen** |

Weiter: `trade_imb = (buy_vol − sell_vol)/(buy_vol + sell_vol)`, positiv =
Käufer aggressiv. `move_bps = (mid[t0+h]/mid[t0] − 1)·10000`, positiv = gestiegen.

## Speicher: Tag für Tag

Ein Tag Orderbuch ≈ 3,4 GB. Der Runner lädt **einen** Tag, verdichtet ihn zur
kleinen Sekunden-Tabelle (~25 MB), speichert die als Parquet und **verwirft die
Rohdaten sofort** (`del` + `gc.collect()`). Es liegt **nie** mehr als ein Tag
Rohdaten im Speicher. Die Analysen laufen nur auf den kleinen Tabellen.

## Aufbau

- **Stichprobe:** *alle* Zeitpunkte im **Sekundenraster** (nicht nur an Levels).
  Flags `near_level` (nahe VAH/VAL der Vorsession) und `active` (Handel in den
  letzten 30 s) markieren Teilmengen zum Vergleich.
- **Vorher-Merkmale** (Fenster 5/15/30 s, **ausschließlich vor t0**):
  `stack_bid`, `stack_ask`, `pull_bid`, `pull_ask` (je Seite getrennt),
  `imbalance`, `spread`, `depth` (Tiefe der besten Level), `vol`, `trade_imb`.
- **Zielgrößen:** (a) Richtung des Moves, (b) binär: `|move|` über
  **12 bps (Taker)** bzw. **6 bps (Maker)**.
- **Auswertung A (Größe):** unterscheiden sich die Vorher-Merkmale zwischen
  großem und kleinem Folge-Move?
- **Auswertung B (Richtung):** *nur unter den großen Moves* — sagt die
  Vorher-Imbalance das Vorzeichen vorher? Trefferquote **immer neben `base_rate`**
  (= wie gut schon „immer dieselbe Richtung" wäre); ohne diesen Vergleich ist
  eine Trefferquote nicht interpretierbar.
- **Kontrollgruppe:** zufällige Zeitpunkte aus aktiven Phasen, gleiche Messung.
- **Kein Look-ahead:** Fenstersummen laufen über `[t0−w, t0−1]`, also strikt vor
  `t0`; Moves ausschließlich danach.
- **Fallzahlen** je Zelle, `belastbar=False` unter 30. Die **Anzahl aller
  getesteten Varianten** wird am Ende ausgewiesen (~198 Einzelvergleiche
  gepoolt) — einzelne „Treffer" ohne Wiederholung an neuen Tagen sind damit
  nicht belastbar.

## Ergebnisse

Pro Tag **und** gepoolt (damit sichtbar wird, ob ein Effekt über Tage stabil ist
oder nur an einem Tag auftrat). CSVs in `data/output/`:
`dom_move_distribution_{per_day,pooled,subsets}.csv`,
`dom_size_analysis_pooled.csv`,
`dom_direction_analysis_{pooled,per_day,control}.csv`.

---

# Dritte Studie: Volatilitäts-Vorhersage

**Forschungsfrage:** Sagt der Orderbuch-Zustand die realisierte Volatilität der
nächsten 30/60/120 s vorher — **über das hinaus**, was vergangene Volatilität
allein schon vorhersagt?

```bash
python run_vola_study.py --days 5
python run_vola_study.py --days 3 --synth    # offline testen
```

## Sekundentabelle wird persistiert

Ab jetzt speichert die Pipeline die verdichtete **Sekundentabelle** je Tag
(`data/interim/dom_seconds/`, ~10 MB statt 3 GB). Jede weitere Studie läuft
damit **ohne erneuten Download**. Der erste Lauf dieser Studie muss die Tage
einmalig neu holen, danach nicht mehr.

## Zielgrößen (beide ausgewiesen)

| Größe | Definition |
|---|---|
| `rv_<h>` | Standardabweichung der Sekunden-Returns über den Horizont (bps) |
| `range_<h>` | High-Low-Spanne des Mid über den Horizont, relativ zu `mid[t0]` (bps) |

## Merkmale vor t0 (Fenster 5/15/30 s)

`rv_pre` (vergangene Vola), `netout` = `(pull_bid+pull_ask) − (stack_bid+stack_ask)`,
`depth_ratio` (Tiefe Fensterende / Fensteranfang), `depth` (absolut), `spread_bps`,
`churn` = `(pull+stack)/Tiefe`, `vol` (gehandeltes Volumen), `ntrades` (Trade-Anzahl).

## Die Treppe — out-of-sample bewertet

| Stufe | Merkmale |
|---|---|
| **S1** | nur vergangene Volatilität |
| **S2** | S1 + gehandeltes Volumen + Trade-Anzahl |
| **S3** | S2 + Orderbuch (Netto-Abfluss, Tiefenveränderung, Umschichtung, Tiefe, Spread) |

**Wichtig:** In-sample steigt R² *immer*, wenn man Merkmale hinzufügt — S3 > S2
wäre dort garantiert und damit wertlos. Deshalb wird mit
**Leave-one-day-out-Kreuzvalidierung** bewertet: trainieren auf 4 Tagen, testen
auf dem ausgelassenen 5. Nur eine Verbesserung *dort* belegt eigene Information.
Ausgewiesen werden `r2_oos`, `mae_bps` und die Differenz zur **vorigen** Stufe.

## Überlappung und Fallzahl

Bei Sekundenraster und 120 s Horizont teilen sich 120 aufeinanderfolgende Punkte
fast denselben Move. Alle Auswertungen laufen auf **nicht-überlappenden Blöcken**
(Schrittweite = Horizont); ausgewiesen wird die **effektive** Fallzahl, nicht die
Rohzahl (Faktor 30/60/120).

## Terzile nach Netto-Abfluss

Terzilgrenzen werden **innerhalb jedes Tages** bestimmt — gepoolte Grenzen würden
vor allem Wochentage von Wochenenden trennen statt Marktzustände. `netout`
positiv = mehr Liquidität abgezogen als aufgebaut.

Ergebnisse in `data/output/`: `vola_targets.csv`, `vola_ladder.csv`,
`vola_terciles_{pooled,per_day}.csv`.

---

## Nächste Schritte (erst nach Sichtung eines echten Tages)

- **Ein echter Tag zuerst.** `python run_all.py` mit deinem Key laufen lassen,
  die ROH-Spalten aus Schritt 1 und den Report ansehen.
- **Dann erst mehrere Tage.** In `config.py` `START_DATE`/`END_DATE` weiten; die
  Referenz-Logik nutzt dann automatisch die jeweils vorige Session (auch über
  Tagesgrenzen). Bei größeren Zeiträumen wird die Top-of-Book-Rekonstruktion
  (Schritt 4) zur langsamsten Stelle — ggf. tageweise verarbeiten.
```
