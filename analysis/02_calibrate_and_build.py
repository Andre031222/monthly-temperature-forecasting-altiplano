import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from scipy import stats

from config import (
    DATA,
    MIN_DAILY_COVERAGE,
    RANDOM_STATE,
    STATION,
    TRAIN_END,
    WMO_MAX_CONSECUTIVE_GAP,
    WMO_MAX_MISSING_DAYS,
    longest_run_of_missing,
    save_json,
)


def load_daily():
    daily = pd.read_csv(DATA / f"daily_{STATION}.csv", parse_dates=["date"])
    daily["tmid"] = (daily["tmax"] + daily["tmin"]) / 2
    return daily


def evaluate_raw_midpoint(daily):
    paired = daily.dropna(subset=["tmean", "tmid"])
    error = paired["tmid"] - paired["tmean"]
    t_stat, p_value = stats.ttest_1samp(error, 0.0)
    return {
        "n_paired_days": int(len(paired)),
        "bias_C": round(float(error.mean()), 4),
        "mae_C": round(float(error.abs().mean()), 4),
        "rmse_C": round(float(np.sqrt((error ** 2).mean())), 4),
        "pearson_r": round(float(paired["tmid"].corr(paired["tmean"])), 4),
        "paired_t": round(float(t_stat), 2),
        "p_value": float(p_value),
    }


def fit_calibration(daily):
    trainable = daily["tmean"].notna() & daily["tmid"].notna()
    train = trainable & (daily["date"] <= TRAIN_END)
    holdout = trainable & (daily["date"] > TRAIN_END)

    x_train = daily.loc[train, ["tmid"]].to_numpy()
    y_train = daily.loc[train, "tmean"].to_numpy()

    model = LinearRegression().fit(x_train, y_train)
    cv_pred = cross_val_predict(
        LinearRegression(),
        x_train,
        y_train,
        cv=KFold(10, shuffle=True, random_state=RANDOM_STATE),
    )

    x_hold = daily.loc[holdout, ["tmid"]].to_numpy()
    y_hold = daily.loc[holdout, "tmean"].to_numpy()
    y_pred = model.predict(x_hold)

    metrics = {
        "equation": f"Tmean = {model.intercept_:.4f} + {model.coef_[0]:.4f} * (Tmax + Tmin) / 2",
        "intercept": round(float(model.intercept_), 4),
        "slope": round(float(model.coef_[0]), 4),
        "n_fit_days": int(train.sum()),
        "fit_period": f"{daily.loc[train, 'date'].min().date()} .. {daily.loc[train, 'date'].max().date()}",
        "cv10_mae_C": round(float(mean_absolute_error(y_train, cv_pred)), 4),
        "cv10_rmse_C": round(float(np.sqrt(mean_squared_error(y_train, cv_pred))), 4),
        "cv10_r2": round(float(r2_score(y_train, cv_pred)), 4),
        "cv10_bias_C": round(float((cv_pred - y_train).mean()), 5),
        "holdout_n_days": int(holdout.sum()),
        "holdout_mae_C": round(float(mean_absolute_error(y_hold, y_pred)), 4),
        "holdout_rmse_C": round(float(np.sqrt(mean_squared_error(y_hold, y_pred))), 4),
        "holdout_bias_C": round(float((y_pred - y_hold).mean()), 4),
    }
    return model, metrics


def apply_calibration(daily, model):
    daily = daily.copy()
    fillable = daily["tmean"].isna() & daily["tmid"].notna()
    daily["tmean_source"] = np.where(
        daily["tmean"].notna(), "observed", np.where(fillable, "calibrated", "missing")
    )
    daily.loc[fillable, "tmean"] = model.predict(daily.loc[fillable, ["tmid"]].to_numpy())
    return daily, int(fillable.sum())


def to_monthly(daily):
    calendar = pd.DataFrame(
        {"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")}
    )
    frame = calendar.merge(daily, on="date", how="left")
    frame["ym"] = frame["date"].values.astype("datetime64[M]")

    grouped = frame.groupby("ym")
    monthly = grouped.agg(
        tmean=("tmean", "mean"),
        tmax=("tmax", "mean"),
        tmin=("tmin", "mean"),
        rh=("rh", "mean"),
        precip=("precip", "sum"),
        n_tmean=("tmean", "count"),
        n_rh=("rh", "count"),
        n_precip=("precip", "count"),
    ).reset_index()

    monthly["days_expected"] = pd.to_datetime(monthly["ym"]).dt.days_in_month
    monthly["days_missing"] = monthly["days_expected"] - monthly["n_tmean"]
    monthly["longest_gap"] = grouped["tmean"].apply(longest_run_of_missing).values
    monthly["n_calibrated"] = (
        grouped["tmean_source"].apply(lambda s: (s == "calibrated").sum()).values
    )
    monthly["wmo_valid"] = (monthly["days_missing"] <= WMO_MAX_MISSING_DAYS) & (
        monthly["longest_gap"] < WMO_MAX_CONSECUTIVE_GAP
    )

    monthly.loc[~monthly["wmo_valid"], ["tmean", "tmax", "tmin"]] = np.nan
    monthly.loc[monthly["n_rh"] / monthly["days_expected"] < MIN_DAILY_COVERAGE, "rh"] = np.nan
    monthly.loc[monthly["n_precip"] / monthly["days_expected"] < MIN_DAILY_COVERAGE, "precip"] = np.nan

    monthly["dtr"] = monthly["tmax"] - monthly["tmin"]
    monthly["year"] = pd.to_datetime(monthly["ym"]).dt.year
    monthly["month"] = pd.to_datetime(monthly["ym"]).dt.month
    return monthly


def describe_monthly(monthly):
    valid = monthly.dropna(subset=["tmean"])
    series = valid.set_index(pd.to_datetime(valid["ym"]))["tmean"].asfreq("MS")
    time_index = (valid["year"] - valid["year"].min()) * 12 + valid["month"] - 1
    slope, _, _, p_value, std_err = stats.linregress(time_index, valid["tmean"])

    return {
        "months_total": int(len(monthly)),
        "months_valid": int(len(valid)),
        "pct_valid": round(float(len(valid) / len(monthly) * 100), 2),
        "mean_C": round(float(valid["tmean"].mean()), 3),
        "sd_C": round(float(valid["tmean"].std()), 3),
        "min_C": round(float(valid["tmean"].min()), 3),
        "max_C": round(float(valid["tmean"].max()), 3),
        "pct_days_calibrated": round(
            float(valid["n_calibrated"].sum() / valid["days_expected"].sum() * 100), 2
        ),
        "trend_C_per_decade": round(float(slope * 120), 4),
        "trend_p_value": float(p_value),
        "trend_std_err": round(float(std_err * 120), 4),
        "autocorrelation": {
            f"lag_{k}": round(float(series.autocorr(k)), 3) for k in (1, 2, 3, 6, 12, 24)
        },
        "climatology_C": valid.groupby("month")["tmean"].mean().round(3).to_dict(),
        "climatology_sd_C": valid.groupby("month")["tmean"].std().round(3).to_dict(),
    }


def training_climatology(monthly):
    train = monthly[(pd.to_datetime(monthly["ym"]) <= TRAIN_END) & monthly["tmean"].notna()]
    return train.groupby("month")["tmean"].mean()


def build_supervised(monthly, climatology):
    series = monthly.set_index(pd.to_datetime(monthly["ym"])).asfreq("MS")
    features = pd.DataFrame(index=series.index)

    features["target"] = series["tmean"].shift(-1)
    features["tmean_lag0"] = series["tmean"]
    features["tmean_lag1"] = series["tmean"].shift(1)
    features["tmean_lag2"] = series["tmean"].shift(2)
    features["dtr_lag0"] = series["dtr"]
    features["rh_lag0"] = series["rh"]
    features["precip_lag0"] = series["precip"]
    features["roll3"] = series["tmean"].rolling(3).mean()

    target_month = (features.index + pd.offsets.MonthBegin(1)).month
    features["clim_target"] = pd.Series(target_month, index=features.index).map(climatology)
    features["clim_lag0"] = pd.Series(features.index.month, index=features.index).map(climatology)
    features["anomaly_lag0"] = features["tmean_lag0"] - features["clim_lag0"]
    features["sin_month"] = np.sin(2 * np.pi * target_month / 12)
    features["cos_month"] = np.cos(2 * np.pi * target_month / 12)
    features["trend"] = np.arange(len(features))
    features["target_date"] = features.index + pd.offsets.MonthBegin(1)
    features["target_month"] = target_month
    features["persistence_ref"] = features["tmean_lag0"]

    return features.dropna()


def main():
    daily = load_daily()
    report = {"raw_midpoint": evaluate_raw_midpoint(daily)}

    model, report["calibration"] = fit_calibration(daily)
    daily, filled = apply_calibration(daily, model)
    report["calibration"]["days_filled"] = filled

    monthly = to_monthly(daily)
    monthly.to_csv(DATA / "monthly_calibrated.csv", index=False)
    report["monthly_series"] = describe_monthly(monthly)

    climatology = training_climatology(monthly)
    climatology.to_csv(DATA / "climatology_train.csv")

    supervised = build_supervised(monthly, climatology)
    supervised.to_csv(DATA / "supervised_h1.csv")

    excluded = ("target", "target_date", "target_month", "persistence_ref", "clim_lag0")
    predictors = [c for c in supervised.columns if c not in excluded]
    report["supervised_dataset"] = {
        "samples": int(len(supervised)),
        "predictors": predictors,
        "n_predictors": len(predictors),
        "span": f"{supervised['target_date'].min().date()} .. {supervised['target_date'].max().date()}",
        "n_train": int((supervised["target_date"] <= TRAIN_END).sum()),
        "n_test": int((supervised["target_date"] > TRAIN_END).sum()),
    }

    print(save_json(report, "02_calibration.json"))


if __name__ == "__main__":
    main()
