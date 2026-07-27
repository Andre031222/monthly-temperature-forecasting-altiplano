import numpy as np
import pandas as pd

from config import COLUMN_MAP, DATA, STATION, WEATHER_VARS, XLSX, save_json


def read_station(sheet):
    raw = pd.read_excel(XLSX, sheet_name=sheet, skiprows=2, usecols="B:I")
    raw.columns = [COLUMN_MAP.get(c, c) for c in raw.columns]
    return raw[pd.to_numeric(raw["year"], errors="coerce").between(1900, 2100)]


def clean(df):
    df = df.copy()
    for column in ["year", "month", "day"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in WEATHER_VARS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column] <= -99, column] = np.nan

    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce"
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    df.loc[~df["rh"].between(0, 100), "rh"] = np.nan
    return df


def audit(df):
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    absent = calendar.difference(df["date"])
    blocks = []
    if len(absent):
        gaps = pd.Series(absent)
        groups = (gaps.diff() != pd.Timedelta("1D")).cumsum()
        for _, block in gaps.groupby(groups):
            blocks.append(
                {
                    "from": str(block.min().date()),
                    "to": str(block.max().date()),
                    "days": int(len(block)),
                }
            )

    return {
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "rows": int(len(df)),
        "expected_days": int(len(calendar)),
        "absent_days": int(len(absent)),
        "absent_blocks": blocks,
        "missing_pct": {
            v: round(float(df[v].isna().mean() * 100), 2) for v in WEATHER_VARS
        },
        "inconsistent_temperature_rows": int(
            ((df["tmin"] > df["tmean"]) | (df["tmean"] > df["tmax"])).sum()
        ),
    }


def compare_sheets(sheets):
    first, *rest = sheets
    reference = first[1][WEATHER_VARS].reset_index(drop=True)
    duplicates = []
    for name, frame in rest:
        candidate = frame[WEATHER_VARS].reset_index(drop=True)
        if reference.shape == candidate.shape and reference.equals(candidate):
            duplicates.append(name)
    return duplicates


def main():
    sheets = []
    for sheet in pd.ExcelFile(XLSX).sheet_names:
        frame = clean(read_station(sheet))
        frame.to_csv(DATA / f"daily_{sheet}.csv", index=False)
        sheets.append((sheet, frame))

    report = {name: audit(frame) for name, frame in sheets}
    report["duplicate_sheets"] = compare_sheets(sheets)
    report["station_used"] = STATION

    print(save_json(report, "01_data_audit.json"))


if __name__ == "__main__":
    main()
