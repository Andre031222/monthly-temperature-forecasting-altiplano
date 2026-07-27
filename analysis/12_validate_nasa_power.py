import itertools

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA, RESULTS, save_json

VARIABLES = ["tmean", "tmax", "tmin", "rh", "precip"]


def load_nasa():
    frame = pd.read_csv(DATA / "nasa_power_all_stations.csv", parse_dates=["date"])
    for column in VARIABLES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def detect_shared_cells(nasa):
    pivot = nasa.pivot_table(index="date", columns="station", values="tmean")
    stations = list(pivot.columns)

    identical_groups = []
    assigned = set()
    for a in stations:
        if a in assigned:
            continue
        group = [a]
        for b in stations:
            if b == a or b in assigned:
                continue
            if np.allclose(pivot[a].to_numpy(), pivot[b].to_numpy(), equal_nan=True):
                group.append(b)
        if len(group) > 1:
            identical_groups.append(sorted(group))
            assigned.update(group)

    correlations = {}
    for a, b in itertools.combinations(stations, 2):
        correlations[f"{a} vs {b}"] = round(float(pivot[a].corr(pivot[b])), 4)

    return {
        "n_stations_requested": len(stations),
        "identical_series_groups": identical_groups,
        "n_independent_series": len(stations) - sum(len(g) - 1 for g in identical_groups),
        "max_pairwise_correlation": round(float(max(correlations.values())), 4),
        "min_pairwise_correlation": round(float(min(correlations.values())), 4),
        "mean_pairwise_correlation": round(float(np.mean(list(correlations.values()))), 4),
        "pairs_above_0.99": [k for k, v in correlations.items() if v > 0.99],
    }


def compare_with_senamhi(nasa):
    senamhi = pd.read_csv(DATA / "daily_PUNO.csv", parse_dates=["date"])
    puno = nasa[nasa["station"] == "Puno"][["date", "tmean", "tmax", "tmin", "rh", "precip"]]

    merged = senamhi.merge(puno, on="date", suffixes=("_senamhi", "_nasa"))
    results = {"n_matched_days": int(len(merged))}

    for variable in VARIABLES:
        pair = merged[[f"{variable}_senamhi", f"{variable}_nasa"]].dropna()
        if len(pair) < 30:
            continue
        difference = pair.iloc[:, 1] - pair.iloc[:, 0]
        t_stat, p_value = stats.ttest_rel(pair.iloc[:, 1], pair.iloc[:, 0])
        results[variable] = {
            "n": int(len(pair)),
            "senamhi_mean": round(float(pair.iloc[:, 0].mean()), 3),
            "nasa_mean": round(float(pair.iloc[:, 1].mean()), 3),
            "bias_nasa_minus_senamhi": round(float(difference.mean()), 3),
            "mae": round(float(difference.abs().mean()), 3),
            "rmse": round(float(np.sqrt((difference ** 2).mean())), 3),
            "pearson_r": round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 4),
            "paired_t": round(float(t_stat), 2),
            "p_value": float(p_value),
        }

    monthly_senamhi = pd.read_csv(DATA / "monthly_calibrated.csv", parse_dates=["ym"])
    puno_monthly = puno.copy()
    puno_monthly["ym"] = puno_monthly["date"].values.astype("datetime64[M]")
    aggregated = puno_monthly.groupby("ym")["tmean"].mean().reset_index()
    merged_monthly = monthly_senamhi[["ym", "tmean"]].dropna().merge(
        aggregated, on="ym", suffixes=("_senamhi", "_nasa")
    )
    difference = merged_monthly["tmean_nasa"] - merged_monthly["tmean_senamhi"]
    results["monthly_tmean"] = {
        "n_months": int(len(merged_monthly)),
        "bias_C": round(float(difference.mean()), 3),
        "mae_C": round(float(difference.abs().mean()), 3),
        "rmse_C": round(float(np.sqrt((difference ** 2).mean())), 3),
        "pearson_r": round(float(merged_monthly["tmean_senamhi"].corr(merged_monthly["tmean_nasa"])), 4),
        "anomaly_correlation": round(float(
            (merged_monthly["tmean_senamhi"] - merged_monthly["tmean_senamhi"].mean()).corr(
                merged_monthly["tmean_nasa"] - merged_monthly["tmean_nasa"].mean())
        ), 4),
    }
    merged_monthly.to_csv(RESULTS / "nasa_vs_senamhi_monthly.csv", index=False)
    return results


def main():
    nasa = load_nasa()
    report = {
        "independence": detect_shared_cells(nasa),
        "validation_against_senamhi": compare_with_senamhi(nasa),
    }
    print(save_json(report, "12_nasa_validation.json"))


if __name__ == "__main__":
    main()
