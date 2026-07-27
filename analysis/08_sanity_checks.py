import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, RANDOM_STATE, RESULTS, STATION, TRAIN_END, XLSX, save_json

PREDICTORS = [
    "tmean_lag0", "tmean_lag1", "tmean_lag2", "dtr_lag0", "rh_lag0", "precip_lag0",
    "roll3", "clim_target", "anomaly_lag0", "sin_month", "cos_month", "trend",
]


def check_counts():
    raw = pd.read_excel(XLSX, sheet_name=STATION, skiprows=2, usecols="B:I")
    daily = pd.read_csv(DATA / f"daily_{STATION}.csv", parse_dates=["date"])
    monthly = pd.read_csv(DATA / "monthly_calibrated.csv", parse_dates=["ym"])
    supervised = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])

    return {
        "excel_rows_read": int(len(raw)),
        "daily_rows_after_cleaning": int(len(daily)),
        "rows_dropped_as_non_data": int(len(raw) - len(daily)),
        "unique_dates": int(daily["date"].nunique()),
        "duplicate_dates": int(len(daily) - daily["date"].nunique()),
        "monthly_rows": int(len(monthly)),
        "monthly_valid_tmean": int(monthly["tmean"].notna().sum()),
        "supervised_rows": int(len(supervised)),
        "supervised_nan_cells": int(supervised[PREDICTORS + ["target"]].isna().sum().sum()),
    }


def check_split_integrity():
    supervised = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    train = supervised[supervised["target_date"] <= TRAIN_END]
    test = supervised[supervised["target_date"] > TRAIN_END]

    overlap = set(train["target_date"]) & set(test["target_date"])
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "sum_equals_total": bool(len(train) + len(test) == len(supervised)),
        "overlapping_target_dates": int(len(overlap)),
        "last_train_target": str(train["target_date"].max().date()),
        "first_test_target": str(test["target_date"].min().date()),
        "train_strictly_before_test": bool(train["target_date"].max() < test["target_date"].min()),
        "feature_dates_precede_targets": bool(
            (supervised.index < supervised["target_date"]).all()
        ),
    }


def check_predictions_are_distinct():
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["target_date"])
    columns = [c for c in predictions.columns if c not in ("target_date", "observed")]
    pairs = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            identical = bool(np.allclose(predictions[a], predictions[b]))
            pairs[f"{a} vs {b}"] = {
                "identical": identical,
                "max_abs_difference": round(float((predictions[a] - predictions[b]).abs().max()), 4),
            }
    return {
        "n_test_rows": int(len(predictions)),
        "any_prediction_equals_observed": bool(
            any(np.allclose(predictions[c], predictions["observed"]) for c in columns)
        ),
        "pairwise": pairs,
    }


def permutation_leakage_test(n_permutations=200):
    supervised = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    train = supervised[supervised["target_date"] <= TRAIN_END]
    test = supervised[supervised["target_date"] > TRAIN_END]

    model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    model.fit(train[PREDICTORS], train["target"])
    real_mae = mean_absolute_error(test["target"], model.predict(test[PREDICTORS]))

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = []
    for _ in range(n_permutations):
        y = rng.permutation(train["target"].to_numpy())
        permuted = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        permuted.fit(train[PREDICTORS], y)
        shuffled.append(mean_absolute_error(test["target"], permuted.predict(test[PREDICTORS])))

    shuffled = np.array(shuffled)
    return {
        "real_mae_C": round(float(real_mae), 4),
        "shuffled_mae_mean_C": round(float(shuffled.mean()), 4),
        "shuffled_mae_min_C": round(float(shuffled.min()), 4),
        "n_permutations": n_permutations,
        "n_shuffled_better_than_real": int((shuffled <= real_mae).sum()),
        "empirical_p_value": round(float(((shuffled <= real_mae).sum() + 1) / (n_permutations + 1)), 5),
        "interpretation": "If shuffling the target destroys accuracy, the model learned signal, not leakage.",
    }


def check_missing_data_handling():
    daily = pd.read_csv(DATA / f"daily_{STATION}.csv", parse_dates=["date"])
    monthly = pd.read_csv(DATA / "monthly_calibrated.csv", parse_dates=["ym"])
    supervised = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])

    rejected = monthly[monthly["tmean"].isna()]
    return {
        "daily_missing_tmean_before_calibration": int(daily["tmean"].isna().sum()),
        "months_rejected_by_wmo_rule": int(len(rejected)),
        "rejected_months_all_have_gaps": bool((rejected["days_missing"] > 0).all()),
        "max_missing_days_in_an_accepted_month": int(monthly.loc[monthly["tmean"].notna(), "days_missing"].max()),
        "max_consecutive_gap_in_an_accepted_month": int(monthly.loc[monthly["tmean"].notna(), "longest_gap"].max()),
        "wmo_thresholds": "accepted months must have <= 10 missing days and < 5 consecutive",
        "calibrated_days_in_accepted_months": int(monthly.loc[monthly["tmean"].notna(), "n_calibrated"].sum()),
        "supervised_samples_lost_to_gaps": int(236 - 1 - len(supervised)),
        "target_never_reconstructed_beyond_limit": bool(
            monthly.loc[monthly["tmean"].notna(), "n_calibrated"].max() <= monthly["days_expected"].max()
        ),
    }


def measure_runtime():
    supervised = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    train = supervised[supervised["target_date"] <= TRAIN_END]
    start = time.perf_counter()
    for _ in range(100):
        model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        model.fit(train[PREDICTORS], train["target"])
    elapsed = time.perf_counter() - start
    return {
        "matrix_shape": f"{len(train)} rows x {len(PREDICTORS)} columns",
        "cells_in_training_matrix": int(len(train) * len(PREDICTORS)),
        "seconds_for_100_linear_fits": round(elapsed, 4),
        "milliseconds_per_fit": round(elapsed * 10, 3),
        "note": "The dataset is small by design: 22 years of monthly data cannot exceed 264 rows.",
    }


def main():
    report = {
        "record_counts": check_counts(),
        "split_integrity": check_split_integrity(),
        "predictions_distinct": check_predictions_are_distinct(),
        "missing_data_handling": check_missing_data_handling(),
        "permutation_leakage_test": permutation_leakage_test(),
        "runtime": measure_runtime(),
    }
    print(save_json(report, "08_sanity_checks.json"))


if __name__ == "__main__":
    main()
