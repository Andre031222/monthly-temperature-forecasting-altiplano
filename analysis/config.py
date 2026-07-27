from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_SOURCES = [
    ROOT / "data" / "senamhi_puno_2003_2024.xlsx",
    ROOT / "LEONEL COYLA IDME.xlsx",
]
XLSX = next((path for path in CANDIDATE_SOURCES if path.exists()), CANDIDATE_SOURCES[0])

DATA = ROOT / "analysis" / "data"
RESULTS = ROOT / "analysis" / "results"
FIGURES = ROOT / "analysis" / "figures"

for path in (DATA, RESULTS, FIGURES):
    path.mkdir(parents=True, exist_ok=True)

STATION = "PUNO"
STATION_LABEL = "Puno Principal Climatological Station"

TRAIN_END = pd.Timestamp("2016-12-31")
RANDOM_STATE = 42

WMO_MAX_MISSING_DAYS = 10
WMO_MAX_CONSECUTIVE_GAP = 5
MIN_DAILY_COVERAGE = 0.80

COLUMN_MAP = {
    "Año": "year",
    "Mes": "month",
    "Dia": "day",
    "Humedad Relativa Media (%)": "rh",
    "Precipitación Total (mm)": "precip",
    "Temperatura Media (°C)": "tmean",
    "Temperatura Máxima Media (°C)": "tmax",
    "Temperatura Mínima Media (°C)": "tmin",
}

WEATHER_VARS = ["rh", "precip", "tmean", "tmax", "tmin"]


def save_json(payload, filename):
    import json

    (RESULTS / filename).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def longest_run_of_missing(series):
    longest = current = 0
    for is_missing in series.isna().to_numpy():
        current = current + 1 if is_missing else 0
        longest = max(longest, current)
    return longest
