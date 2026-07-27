import io
import time
import urllib.request

import pandas as pd

from config import DATA, save_json

STATIONS = {
    "Puno": (-15.84, -70.02, 3825),
    "Juliaca": (-15.49, -70.13, 3824),
    "Azangaro": (-14.89, -70.10, 3859),
    "Ayaviri": (-15.25, -69.29, 3918),
    "Macusani": (-14.08333, -70.43333, 4345),
    "Mazocruz": (-16.74250, -69.71611, 3990),
    "Lampa": (-15.35, -70.36667, 3872),
    "Yunguyo": (-16.25, -69.08333, 3826),
    "Juli": (-16.21667, -69.45, 3812),
    "Desaguadero": (-16.56556, -69.04167, 3808),
    "Cojata": (-15.02, -69.37, 4320),
    "Crucero": (-14.34538, -70.03210, 4130),
    "Crucero_Alto": (-15.77948, -70.91799, 4470),
}

PARAMETERS = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,PS"
START = "20000101"
END = "20241231"
OUT = DATA / "nasa_power"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(latitude, longitude):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={PARAMETERS}&community=AG"
        f"&latitude={latitude}&longitude={longitude}"
        f"&start={START}&end={END}&format=CSV"
    )
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read().decode("utf-8")


def parse(text):
    lines = text.splitlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip() == "-END HEADER-")
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_end + 1:])))
    frame = frame.replace(-999.0, pd.NA)
    frame["date"] = pd.to_datetime(frame["YEAR"].astype(str), format="%Y") + pd.to_timedelta(
        frame["DOY"] - 1, unit="D"
    )
    return frame.rename(columns={
        "T2M": "tmean", "T2M_MAX": "tmax", "T2M_MIN": "tmin",
        "RH2M": "rh", "PRECTOTCORR": "precip", "WS2M": "wind", "PS": "pressure",
    })[["date", "tmean", "tmax", "tmin", "rh", "precip", "wind", "pressure"]]


def main():
    report = {}
    frames = []

    for name, (latitude, longitude, elevation) in STATIONS.items():
        frame = parse(fetch(latitude, longitude))
        frame.insert(0, "station", name)
        frame["latitude"] = latitude
        frame["longitude"] = longitude
        frame["elevation"] = elevation
        frame.to_csv(OUT / f"{name}.csv", index=False)
        frames.append(frame)

        report[name] = {
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation,
            "rows": int(len(frame)),
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "missing_tmean": int(frame["tmean"].isna().sum()),
            "mean_tmean_C": round(float(pd.to_numeric(frame["tmean"]).mean()), 3),
        }
        print(f"[{name:14s}] {report[name]['rows']} rows | "
              f"missing tmean: {report[name]['missing_tmean']} | mean: {report[name]['mean_tmean_C']} C")
        time.sleep(2)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(DATA / "nasa_power_all_stations.csv", index=False)
    report["_summary"] = {
        "stations": len(STATIONS),
        "total_rows": int(len(combined)),
        "period": f"{START} .. {END}",
        "source": "NASA POWER / MERRA-2, https://power.larc.nasa.gov",
    }

    print(save_json(report, "11_nasa_power_download.json"))


if __name__ == "__main__":
    main()
