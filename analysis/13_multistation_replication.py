import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, RANDOM_STATE, RESULTS, save_json

DUPLICATE_STATIONS = ["Cojata", "Yunguyo"]
TRAIN_END = pd.Timestamp("2018-12-31")

PREDICTORS = [
    "tmean_lag0", "tmean_lag1", "tmean_lag2", "dtr_lag0", "rh_lag0", "precip_lag0",
    "roll3", "clim_target", "anomaly_lag0", "sin_month", "cos_month", "trend",
]

MODELS = ["Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"]


def build_model(name, seed=RANDOM_STATE):
    if name == "Multiple Linear Regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "Random Forest":
        return Pipeline([("model", RandomForestRegressor(
            n_estimators=300, max_depth=6, max_features="sqrt", random_state=seed))])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, learning_rate_init=0.01,
                               max_iter=8000, random_state=seed)),
    ])


def monthly_series(daily):
    daily = daily.copy()
    daily["ym"] = daily["date"].values.astype("datetime64[M]")
    monthly = daily.groupby("ym").agg(
        tmean=("tmean", "mean"), tmax=("tmax", "mean"), tmin=("tmin", "mean"),
        rh=("rh", "mean"), precip=("precip", "sum"),
    ).reset_index()
    monthly["dtr"] = monthly["tmax"] - monthly["tmin"]
    monthly["month"] = pd.to_datetime(monthly["ym"]).dt.month
    return monthly


def supervised_matrix(monthly):
    climatology = monthly[pd.to_datetime(monthly["ym"]) <= TRAIN_END].groupby("month")["tmean"].mean()
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
    features["anomaly_lag0"] = features["tmean_lag0"] - pd.Series(
        features.index.month, index=features.index).map(climatology)
    features["sin_month"] = np.sin(2 * np.pi * target_month / 12)
    features["cos_month"] = np.cos(2 * np.pi * target_month / 12)
    features["trend"] = np.arange(len(features))
    features["target_date"] = features.index + pd.offsets.MonthBegin(1)
    features["persistence_ref"] = features["tmean_lag0"]

    return features.dropna()


def score(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate_station(dataset):
    train = dataset[dataset["target_date"] <= TRAIN_END]
    test = dataset[dataset["target_date"] > TRAIN_END]
    if len(train) < 60 or len(test) < 24:
        return None

    results = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "Climatology": score(test["target"], test["clim_target"]),
        "Persistence": score(test["target"], test["persistence_ref"]),
    }
    for name in MODELS:
        model = build_model(name)
        model.fit(train[PREDICTORS], train["target"])
        results[name] = score(test["target"], model.predict(test[PREDICTORS]))
    return results


def pooled_learning_curve(pooled_train, pooled_test, sizes, seeds=(0, 1, 2)):
    rng = np.random.default_rng(RANDOM_STATE)
    curve = {name: [] for name in MODELS}
    for size in sizes:
        if size > len(pooled_train):
            continue
        for name in MODELS:
            errors = []
            active_seeds = seeds if name != "Multiple Linear Regression" else (RANDOM_STATE,)
            for seed in active_seeds:
                indices = rng.choice(len(pooled_train), size=size, replace=False)
                subset = pooled_train.iloc[indices]
                model = build_model(name, seed)
                model.fit(subset[PREDICTORS], subset["target"])
                errors.append(mean_absolute_error(pooled_test["target"],
                                                  model.predict(pooled_test[PREDICTORS])))
            curve[name].append({"n_train": int(size),
                                "MAE": round(float(np.mean(errors)), 4),
                                "sd": round(float(np.std(errors)), 4)})
    return curve


def seed_stability_pooled(pooled_train, pooled_test, n_seeds=20):
    linear = build_model("Multiple Linear Regression")
    linear.fit(pooled_train[PREDICTORS], pooled_train["target"])
    linear_mae = mean_absolute_error(pooled_test["target"], linear.predict(pooled_test[PREDICTORS]))

    results = {"Multiple Linear Regression": {"MAE": round(float(linear_mae), 4), "deterministic": True}}
    for name in ("Random Forest", "Multilayer Perceptron"):
        errors = []
        for seed in range(n_seeds):
            model = build_model(name, seed)
            model.fit(pooled_train[PREDICTORS], pooled_train["target"])
            errors.append(mean_absolute_error(pooled_test["target"], model.predict(pooled_test[PREDICTORS])))
        errors = np.array(errors)
        results[name] = {
            "n_seeds": n_seeds,
            "mean_MAE": round(float(errors.mean()), 4),
            "sd_MAE": round(float(errors.std(ddof=1)), 4),
            "min_MAE": round(float(errors.min()), 4),
            "max_MAE": round(float(errors.max()), 4),
            "seeds_beating_linear": int((errors < linear_mae).sum()),
            "share_beating_linear": round(float((errors < linear_mae).mean()), 3),
        }
    return results


def main():
    nasa = pd.read_csv(DATA / "nasa_power_all_stations.csv", parse_dates=["date"])
    for column in ["tmean", "tmax", "tmin", "rh", "precip"]:
        nasa[column] = pd.to_numeric(nasa[column], errors="coerce")
    nasa = nasa[~nasa["station"].isin(DUPLICATE_STATIONS)]

    per_station = {}
    datasets = []
    for station, group in nasa.groupby("station"):
        dataset = supervised_matrix(monthly_series(group))
        dataset["station"] = station
        datasets.append(dataset)
        evaluation = evaluate_station(dataset)
        if evaluation:
            per_station[station] = evaluation

    pooled = pd.concat(datasets, ignore_index=False)
    pooled_train = pooled[pooled["target_date"] <= TRAIN_END]
    pooled_test = pooled[pooled["target_date"] > TRAIN_END]

    pooled_results = {"n_train": int(len(pooled_train)), "n_test": int(len(pooled_test)),
                      "Climatology": score(pooled_test["target"], pooled_test["clim_target"]),
                      "Persistence": score(pooled_test["target"], pooled_test["persistence_ref"])}
    for name in MODELS:
        model = build_model(name)
        model.fit(pooled_train[PREDICTORS], pooled_train["target"])
        pooled_results[name] = score(pooled_test["target"], model.predict(pooled_test[PREDICTORS]))

    sizes = [100, 200, 400, 700, 1000, 1400, 1800, len(pooled_train)]
    report = {
        "design": {
            "source": "NASA POWER / MERRA-2",
            "stations_used": sorted(nasa["station"].unique().tolist()),
            "excluded_as_duplicates": DUPLICATE_STATIONS,
            "train_period": f"2000-01 .. {TRAIN_END.date()}",
            "test_period": "2019-01 .. 2024-12",
        },
        "per_station": per_station,
        "pooled": pooled_results,
        "pooled_learning_curve": pooled_learning_curve(pooled_train, pooled_test, sizes),
        "pooled_seed_stability": seed_stability_pooled(pooled_train, pooled_test),
    }

    rows = []
    for station, entry in sorted(per_station.items()):
        rows.append({
            "Station": station,
            "n test": entry["n_test"],
            "Clim.": f"{entry['Climatology']['MAE']:.3f}",
            "MLR": f"{entry['Multiple Linear Regression']['MAE']:.3f}",
            "RF": f"{entry['Random Forest']['MAE']:.3f}",
            "MLP": f"{entry['Multilayer Perceptron']['MAE']:.3f}",
        })
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {
        "Station": "Pooled", "n test": pooled_results["n_test"],
        "Clim.": f"{pooled_results['Climatology']['MAE']:.3f}",
        "MLR": f"{pooled_results['Multiple Linear Regression']['MAE']:.3f}",
        "RF": f"{pooled_results['Random Forest']['MAE']:.3f}",
        "MLP": f"{pooled_results['Multilayer Perceptron']['MAE']:.3f}",
    }
    frame.to_csv(RESULTS / "table12_multistation.csv", index=False)

    print(save_json(report, "13_multistation_replication.json"))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
