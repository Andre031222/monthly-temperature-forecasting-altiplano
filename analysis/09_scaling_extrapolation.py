import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, RANDOM_STATE, RESULTS, TRAIN_END, save_json

PREDICTORS = [
    "tmean_lag0", "tmean_lag1", "tmean_lag2", "dtr_lag0", "rh_lag0", "precip_lag0",
    "roll3", "clim_target", "anomaly_lag0", "sin_month", "cos_month", "trend",
]

SIZES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 137]
SEEDS = [0, 1, 2, 3, 4]
MODELS = ["Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"]


def build_model(name, seed):
    if name == "Multiple Linear Regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "Random Forest":
        return Pipeline([("model", RandomForestRegressor(
            n_estimators=300, max_depth=6, max_features="sqrt", min_samples_leaf=1, random_state=seed))])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, learning_rate_init=0.01,
                               max_iter=8000, random_state=seed)),
    ])


def windows_of_size(n_train, size, max_windows=6):
    if size >= n_train:
        return [(0, n_train)]
    starts = np.linspace(0, n_train - size, min(max_windows, n_train - size + 1)).astype(int)
    return [(int(s), int(s + size)) for s in np.unique(starts)]


def measure_curves(train, test):
    records = []
    for size in SIZES:
        for start, end in windows_of_size(len(train), size):
            subset = train.iloc[start:end]
            for name in MODELS:
                seeds = SEEDS if name != "Multiple Linear Regression" else [RANDOM_STATE]
                for seed in seeds:
                    model = build_model(name, seed)
                    model.fit(subset[PREDICTORS], subset["target"])
                    mae = mean_absolute_error(test["target"], model.predict(test[PREDICTORS]))
                    records.append({"model": name, "n_train": size, "start": start,
                                    "seed": seed, "mae": mae})
    return pd.DataFrame(records)


def power_law(n, a, b, c):
    return a * np.power(n, -b) + c


def fit_scaling(curve):
    grouped = curve.groupby("n_train")["mae"].agg(["mean", "std", "count"]).reset_index()
    x = grouped["n_train"].to_numpy(dtype=float)
    y = grouped["mean"].to_numpy()
    try:
        params, covariance = curve_fit(power_law, x, y, p0=[5.0, 0.5, 0.4],
                                       bounds=([0, 0.01, 0.0], [1e4, 3.0, 1.5]), maxfev=20000)
        errors = np.sqrt(np.diag(covariance))
        residuals = y - power_law(x, *params)
        r_squared = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    except Exception:
        return None, grouped
    return {
        "a": round(float(params[0]), 4),
        "b": round(float(params[1]), 4),
        "asymptote_C": round(float(params[2]), 4),
        "asymptote_se_C": round(float(errors[2]), 4),
        "fit_r2": round(float(r_squared), 4),
    }, grouped


def bootstrap_crossing(curve_a, curve_b, n_boot=500, horizon=600):
    rng = np.random.default_rng(RANDOM_STATE)
    grid = np.arange(140, horizon + 1, 5, dtype=float)
    crossings = []
    for _ in range(n_boot):
        fits = []
        for curve in (curve_a, curve_b):
            sample = curve.groupby("n_train", group_keys=False).apply(
                lambda g: g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
            )
            grouped = sample.groupby("n_train")["mae"].mean().reset_index()
            try:
                params, _ = curve_fit(power_law, grouped["n_train"].to_numpy(dtype=float),
                                      grouped["mae"].to_numpy(), p0=[5.0, 0.5, 0.4],
                                      bounds=([0, 0.01, 0.0], [1e4, 3.0, 1.5]), maxfev=20000)
                fits.append(params)
            except Exception:
                fits.append(None)
        if any(f is None for f in fits):
            continue
        ya, yb = power_law(grid, *fits[0]), power_law(grid, *fits[1])
        below = np.where(yb < ya)[0]
        crossings.append(float(grid[below[0]]) if len(below) else np.nan)

    crossings = np.array(crossings, dtype=float)
    finite = crossings[np.isfinite(crossings)]
    return {
        "n_bootstrap": int(len(crossings)),
        "share_with_crossing_before_horizon": round(float(len(finite) / max(len(crossings), 1)), 3),
        "median_crossing_months": None if not len(finite) else round(float(np.median(finite)), 1),
        "ci95_crossing_months": None if len(finite) < 20 else [
            round(float(np.percentile(finite, 2.5)), 1), round(float(np.percentile(finite, 97.5)), 1)
        ],
        "horizon_months": horizon,
    }


def main():
    data = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    train = data[data["target_date"] <= TRAIN_END]
    test = data[data["target_date"] > TRAIN_END]

    curve = measure_curves(train, test)
    curve.to_csv(RESULTS / "scaling_curves_raw.csv", index=False)

    report = {"design": {
        "sizes": SIZES,
        "windows_per_size": "up to 6 contiguous training windows",
        "seeds_per_window": len(SEEDS),
        "total_fits": int(len(curve)),
    }, "scaling_fits": {}, "measured": {}}

    curves = {}
    for name in MODELS:
        subset = curve[curve["model"] == name]
        curves[name] = subset
        fit, grouped = fit_scaling(subset)
        report["scaling_fits"][name] = fit
        report["measured"][name] = [
            {"n_train": int(row["n_train"]), "mae_mean": round(float(row["mean"]), 4),
             "mae_sd": round(float(row["std"]), 4) if not np.isnan(row["std"]) else None,
             "n_fits": int(row["count"])}
            for _, row in grouped.iterrows()
        ]

    report["crossing_mlr_vs_mlp"] = bootstrap_crossing(
        curves["Multiple Linear Regression"], curves["Multilayer Perceptron"]
    )

    summary = []
    for name in MODELS:
        grouped = curves[name].groupby("n_train")["mae"].mean()
        summary.append({
            "Model": {"Multiple Linear Regression": "MLR", "Random Forest": "RF",
                      "Multilayer Perceptron": "MLP"}[name],
            "MAE at n=30": f"{grouped.loc[30]:.3f}",
            "MAE at n=70": f"{grouped.loc[70]:.3f}",
            "MAE at n=137": f"{grouped.loc[137]:.3f}",
            "Fitted asymptote (°C)": "n/a" if report["scaling_fits"][name] is None
            else f"{report['scaling_fits'][name]['asymptote_C']:.3f} ± {report['scaling_fits'][name]['asymptote_se_C']:.3f}",
            "Fit R²": "n/a" if report["scaling_fits"][name] is None
            else f"{report['scaling_fits'][name]['fit_r2']:.3f}",
        })
    pd.DataFrame(summary).to_csv(RESULTS / "table10_scaling.csv", index=False)

    print(save_json(report, "09_scaling_extrapolation.json"))


if __name__ == "__main__":
    main()
