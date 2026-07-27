import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, RANDOM_STATE, RESULTS, TRAIN_END, save_json

PREDICTORS = [
    "tmean_lag0", "tmean_lag1", "tmean_lag2", "dtr_lag0", "rh_lag0", "precip_lag0",
    "roll3", "clim_target", "anomaly_lag0", "sin_month", "cos_month", "trend",
]

PREDICTOR_GROUPS = {
    "Seasonal only": ["clim_target", "sin_month", "cos_month"],
    "Thermal memory only": ["tmean_lag0", "tmean_lag1", "tmean_lag2", "roll3"],
    "Seasonal + memory": ["clim_target", "sin_month", "cos_month", "tmean_lag0",
                          "tmean_lag1", "tmean_lag2", "roll3", "anomaly_lag0"],
    "Without atmospheric covariates": [p for p in PREDICTORS if p not in ("rh_lag0", "precip_lag0", "dtr_lag0")],
    "All predictors": PREDICTORS,
}

SEASONS = {
    "Wet season (Dec-Mar)": [12, 1, 2, 3],
    "Dry season (May-Aug)": [5, 6, 7, 8],
    "Transition (Apr, Sep-Nov)": [4, 9, 10, 11],
}

BEST_PARAMS = {
    "Multiple Linear Regression": {},
    "Random Forest": {"max_depth": 6, "max_features": "sqrt", "min_samples_leaf": 1, "n_estimators": 300},
    "Multilayer Perceptron": {"alpha": 1.0, "hidden_layer_sizes": (16,), "learning_rate_init": 0.01},
}


def build_model(name):
    if name == "Multiple Linear Regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "Random Forest":
        return Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, **BEST_PARAMS[name]))])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(random_state=RANDOM_STATE, max_iter=8000, **BEST_PARAMS[name])),
    ])


def load_split():
    data = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    return data[data["target_date"] <= TRAIN_END], data[data["target_date"] > TRAIN_END]


def fit_predict(name, train, test, predictors):
    model = build_model(name)
    model.fit(train[predictors], train["target"])
    return model.predict(test[predictors])


def ablation(train, test):
    results = {}
    for label, predictors in PREDICTOR_GROUPS.items():
        entry = {"n_predictors": len(predictors)}
        for name in BEST_PARAMS:
            prediction = fit_predict(name, train, test, predictors)
            entry[name] = {
                "MAE": round(float(mean_absolute_error(test["target"], prediction)), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(test["target"], prediction))), 4),
            }
        results[label] = entry
    return results


def seasonal_breakdown(test, predictions):
    results = {}
    for label, months in SEASONS.items():
        mask = test["target_month"].isin(months).to_numpy()
        entry = {"n_months": int(mask.sum()),
                 "observed_sd_C": round(float(test.loc[mask, "target"].std()), 3)}
        for name, prediction in predictions.items():
            entry[name] = {
                "MAE": round(float(mean_absolute_error(test["target"][mask], prediction[mask])), 4),
                "MBE": round(float(np.mean(prediction[mask] - test["target"][mask].to_numpy())), 4),
            }
        results[label] = entry
    return results


def residual_diagnostics(test, predictions):
    results = {}
    for name, prediction in predictions.items():
        residuals = prediction - test["target"].to_numpy()
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        lag1 = float(pd.Series(residuals).autocorr(1))
        n = len(residuals)
        ljung_stat = n * (n + 2) * sum(
            (pd.Series(residuals).autocorr(k) ** 2) / (n - k) for k in range(1, 6)
        )
        results[name] = {
            "mean_C": round(float(residuals.mean()), 4),
            "sd_C": round(float(residuals.std(ddof=1)), 4),
            "shapiro_W": round(float(shapiro_stat), 4),
            "shapiro_p": round(float(shapiro_p), 4),
            "normal_at_0.05": bool(shapiro_p > 0.05),
            "lag1_autocorrelation": round(lag1, 4),
            "ljung_box_Q5": round(float(ljung_stat), 3),
            "ljung_box_p": round(float(1 - stats.chi2.cdf(ljung_stat, df=5)), 4),
            "white_noise_at_0.05": bool(1 - stats.chi2.cdf(ljung_stat, df=5) > 0.05),
        }
    return results


def learning_curve(train, test):
    sizes = [30, 45, 60, 75, 90, 105, 120, len(train)]
    curve = {name: [] for name in BEST_PARAMS}
    for size in sizes:
        subset = train.tail(size)
        for name in BEST_PARAMS:
            prediction = fit_predict(name, subset, test, PREDICTORS)
            curve[name].append({
                "n_train": int(size),
                "MAE": round(float(mean_absolute_error(test["target"], prediction)), 4),
            })
    return {"sizes": sizes, "curves": curve}


def main():
    train, test = load_split()
    predictions = {name: fit_predict(name, train, test, PREDICTORS) for name in BEST_PARAMS}

    report = {
        "ablation": ablation(train, test),
        "seasonal": seasonal_breakdown(test, predictions),
        "residuals": residual_diagnostics(test, predictions),
        "learning_curve": learning_curve(train, test),
    }

    ablation_rows = []
    for label, entry in report["ablation"].items():
        row = {"Predictor set": label, "k": entry["n_predictors"]}
        for name in BEST_PARAMS:
            row[{"Multiple Linear Regression": "MLR", "Random Forest": "RF",
                 "Multilayer Perceptron": "MLP"}[name]] = f"{entry[name]['MAE']:.3f}"
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(RESULTS / "table7_ablation.csv", index=False)

    seasonal_rows = []
    for label, entry in report["seasonal"].items():
        row = {"Season": label, "n": entry["n_months"], "SD (°C)": f"{entry['observed_sd_C']:.2f}"}
        for name in BEST_PARAMS:
            row[{"Multiple Linear Regression": "MLR", "Random Forest": "RF",
                 "Multilayer Perceptron": "MLP"}[name]] = f"{entry[name]['MAE']:.3f}"
        seasonal_rows.append(row)
    pd.DataFrame(seasonal_rows).to_csv(RESULTS / "table8_seasonal.csv", index=False)

    residual_rows = []
    for name, entry in report["residuals"].items():
        residual_rows.append({
            "Model": {"Multiple Linear Regression": "MLR", "Random Forest": "RF",
                      "Multilayer Perceptron": "MLP"}[name],
            "Mean (°C)": f"{entry['mean_C']:+.3f}",
            "SD (°C)": f"{entry['sd_C']:.3f}",
            "Shapiro-Wilk W": f"{entry['shapiro_W']:.3f}",
            "W p": f"{entry['shapiro_p']:.3f}",
            "Lag-1 r": f"{entry['lag1_autocorrelation']:+.3f}",
            "Ljung-Box Q": f"{entry['ljung_box_Q5']:.2f}",
            "Q p": f"{entry['ljung_box_p']:.3f}",
        })
    pd.DataFrame(residual_rows).to_csv(RESULTS / "table9_residuals.csv", index=False)

    curve_frame = pd.DataFrame({"n_train": report["learning_curve"]["sizes"]})
    for name, points in report["learning_curve"]["curves"].items():
        curve_frame[name] = [point["MAE"] for point in points]
    curve_frame.to_csv(RESULTS / "learning_curve.csv", index=False)

    print(save_json(report, "06_extended_analysis.json"))


if __name__ == "__main__":
    main()
