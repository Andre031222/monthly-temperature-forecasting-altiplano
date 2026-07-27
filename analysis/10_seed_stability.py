import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, RESULTS, TRAIN_END, save_json

PREDICTORS = [
    "tmean_lag0", "tmean_lag1", "tmean_lag2", "dtr_lag0", "rh_lag0", "precip_lag0",
    "roll3", "clim_target", "anomaly_lag0", "sin_month", "cos_month", "trend",
]

N_SEEDS = 50


def make_mlp(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, learning_rate_init=0.01,
                               max_iter=8000, random_state=seed)),
    ])


def make_rf(seed):
    return Pipeline([("model", RandomForestRegressor(
        n_estimators=300, max_depth=6, max_features="sqrt", min_samples_leaf=1, random_state=seed))])


def summarise(errors, reported):
    errors = np.array(errors)
    return {
        "n_seeds": int(len(errors)),
        "mean_MAE": round(float(errors.mean()), 4),
        "sd_MAE": round(float(errors.std(ddof=1)), 4),
        "min_MAE": round(float(errors.min()), 4),
        "max_MAE": round(float(errors.max()), 4),
        "median_MAE": round(float(np.median(errors)), 4),
        "range_MAE": round(float(errors.max() - errors.min()), 4),
        "reported_seed42_MAE": round(float(reported), 4),
        "percentile_of_seed42": round(float((errors < reported).mean() * 100), 1),
    }


def main():
    data = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    train = data[data["target_date"] <= TRAIN_END]
    test = data[data["target_date"] > TRAIN_END]

    linear = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    linear.fit(train[PREDICTORS], train["target"])
    linear_prediction = linear.predict(test[PREDICTORS])
    linear_mae = mean_absolute_error(test["target"], linear_prediction)

    results = {"Multiple Linear Regression": {
        "deterministic": True,
        "MAE": round(float(linear_mae), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(test["target"], linear_prediction))), 4),
        "note": "Ordinary least squares has a closed-form solution; no random seed is involved.",
    }}

    for label, factory in (("Random Forest", make_rf), ("Multilayer Perceptron", make_mlp)):
        errors, rmses, predictions = [], [], []
        for seed in range(N_SEEDS):
            model = factory(seed)
            model.fit(train[PREDICTORS], train["target"])
            prediction = model.predict(test[PREDICTORS])
            predictions.append(prediction)
            errors.append(mean_absolute_error(test["target"], prediction))
            rmses.append(np.sqrt(mean_squared_error(test["target"], prediction)))

        reference = factory(42)
        reference.fit(train[PREDICTORS], train["target"])
        reported_mae = mean_absolute_error(test["target"], reference.predict(test[PREDICTORS]))

        entry = summarise(errors, reported_mae)
        entry["mean_RMSE"] = round(float(np.mean(rmses)), 4)
        entry["sd_RMSE"] = round(float(np.std(rmses, ddof=1)), 4)

        ensemble = np.mean(predictions, axis=0)
        entry["ensemble_MAE"] = round(float(mean_absolute_error(test["target"], ensemble)), 4)
        entry["ensemble_RMSE"] = round(float(np.sqrt(mean_squared_error(test["target"], ensemble))), 4)
        entry["seeds_beating_linear"] = int((np.array(errors) < linear_mae).sum())
        entry["share_beating_linear"] = round(float((np.array(errors) < linear_mae).mean()), 3)
        results[label] = entry

    results["interpretation"] = {
        "linear_mae": round(float(linear_mae), 4),
        "question": "Is the single-seed advantage of the neural model reproducible?",
    }

    rows = []
    for label in ("Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"):
        entry = results[label]
        short = {"Multiple Linear Regression": "MLR", "Random Forest": "RF",
                 "Multilayer Perceptron": "MLP"}[label]
        if entry.get("deterministic"):
            rows.append({"Model": short, "MAE (single fit)": f"{entry['MAE']:.3f}",
                         "Mean MAE over seeds": "deterministic", "SD": "—",
                         "Range over seeds": "—", "Seeds beating MLR": "—"})
        else:
            rows.append({
                "Model": short,
                "MAE (single fit)": f"{entry['reported_seed42_MAE']:.3f}",
                "Mean MAE over seeds": f"{entry['mean_MAE']:.3f}",
                "SD": f"{entry['sd_MAE']:.3f}",
                "Range over seeds": f"{entry['min_MAE']:.3f}–{entry['max_MAE']:.3f}",
                "Seeds beating MLR": f"{entry['seeds_beating_linear']}/{N_SEEDS}",
            })
    pd.DataFrame(rows).to_csv(RESULTS / "table11_seed_stability.csv", index=False)

    print(save_json(results, "10_seed_stability.json"))


if __name__ == "__main__":
    main()
