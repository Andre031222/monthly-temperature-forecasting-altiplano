import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA, RANDOM_STATE, RESULTS, TRAIN_END, save_json

PREDICTORS = [
    "tmean_lag0",
    "tmean_lag1",
    "tmean_lag2",
    "dtr_lag0",
    "rh_lag0",
    "precip_lag0",
    "roll3",
    "clim_target",
    "anomaly_lag0",
    "sin_month",
    "cos_month",
    "trend",
]

SEARCH_SPACES = {
    "Multiple Linear Regression": (
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        {},
    ),
    "Random Forest": (
        Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE))]),
        {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 6, 12],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, 1.0],
        },
    ),
    "Multilayer Perceptron": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        random_state=RANDOM_STATE,
                        max_iter=8000,
                        early_stopping=False,
                    ),
                ),
            ]
        ),
        {
            "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__learning_rate_init": [0.001, 0.01],
        },
    ),
}


def load_dataset():
    data = pd.read_csv(DATA / "supervised_h1.csv", index_col=0, parse_dates=[0, "target_date"])
    train = data[data["target_date"] <= TRAIN_END]
    test = data[data["target_date"] > TRAIN_END]
    return data, train, test


def score(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MBE": round(float(np.mean(np.asarray(y_pred) - np.asarray(y_true))), 4),
    }


def skill_score(y_true, y_pred, y_reference):
    mse_model = mean_squared_error(y_true, y_pred)
    mse_reference = mean_squared_error(y_true, y_reference)
    return round(float(1 - mse_model / mse_reference), 4)


def diebold_mariano(y_true, pred_a, pred_b, power=2):
    loss_a = np.abs(np.asarray(y_true) - np.asarray(pred_a)) ** power
    loss_b = np.abs(np.asarray(y_true) - np.asarray(pred_b)) ** power
    diff = loss_a - loss_b
    n = len(diff)
    statistic = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=n - 1))
    return round(float(statistic), 3), float(p_value)


def bootstrap_interval(y_true, y_pred, metric, n_boot=2000):
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        values.append(metric(y_true[idx], y_pred[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def run_baselines(train, test):
    predictions = {
        "Climatology": test["clim_target"].to_numpy(),
        "Persistence": test["persistence_ref"].to_numpy(),
    }
    train_predictions = {
        "Climatology": train["clim_target"].to_numpy(),
        "Persistence": train["persistence_ref"].to_numpy(),
    }
    return predictions, train_predictions


def run_models(train, test):
    splitter = TimeSeriesSplit(n_splits=5)
    x_train, y_train = train[PREDICTORS], train["target"]
    x_test = test[PREDICTORS]

    predictions = {}
    train_predictions = {}
    tuning = {}
    fitted = {}

    for name, (pipeline, grid) in SEARCH_SPACES.items():
        search = GridSearchCV(
            pipeline,
            grid,
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted[name] = search.best_estimator_
        predictions[name] = search.best_estimator_.predict(x_test)
        train_predictions[name] = search.best_estimator_.predict(x_train)
        tuning[name] = {
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_mae_C": round(float(-search.best_score_), 4),
            "n_candidates": int(len(search.cv_results_["params"])),
        }

    return predictions, train_predictions, tuning, fitted


def importance_report(fitted, test):
    x_test, y_test = test[PREDICTORS], test["target"]
    report = {}

    linear = fitted["Multiple Linear Regression"].named_steps["model"]
    report["linear_coefficients"] = {
        name: round(float(coef), 4) for name, coef in zip(PREDICTORS, linear.coef_)
    }

    for name in ("Random Forest", "Multilayer Perceptron"):
        result = permutation_importance(
            fitted[name], x_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="neg_mean_absolute_error"
        )
        ranked = sorted(
            zip(PREDICTORS, result.importances_mean, result.importances_std),
            key=lambda item: item[1],
            reverse=True,
        )
        report[f"permutation_importance_{name.replace(' ', '_').lower()}"] = [
            {"feature": f, "mean": round(float(m), 4), "sd": round(float(s), 4)} for f, m, s in ranked
        ]

    return report


def main():
    data, train, test = load_dataset()
    y_train, y_test = train["target"], test["target"]

    baseline_test, baseline_train = run_baselines(train, test)
    model_test, model_train, tuning, fitted = run_models(train, test)

    all_test = {**baseline_test, **model_test}
    all_train = {**baseline_train, **model_train}
    reference = baseline_test["Climatology"]

    table = {}
    for name, pred in all_test.items():
        entry = {
            "test": score(y_test, pred),
            "train": score(y_train, all_train[name]),
            "skill_vs_climatology": skill_score(y_test, pred, reference),
            "mae_ci95": bootstrap_interval(y_test, pred, mean_absolute_error),
            "rmse_ci95": bootstrap_interval(
                y_test, pred, lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
            ),
        }
        if name != "Climatology":
            statistic, p_value = diebold_mariano(y_test, pred, reference)
            entry["diebold_mariano_vs_climatology"] = {
                "statistic": statistic,
                "p_value": round(float(p_value), 5),
                "better_than_climatology": bool(statistic < 0 and p_value < 0.05),
            }
        table[name] = entry

    best = min(model_test, key=lambda n: table[n]["test"]["MAE"])
    for name, pred in model_test.items():
        if name != best:
            statistic, p_value = diebold_mariano(y_test, model_test[best], pred)
            table[best].setdefault("diebold_mariano_vs_models", {})[name] = {
                "statistic": statistic,
                "p_value": round(float(p_value), 5),
            }

    predictions = pd.DataFrame({"target_date": test["target_date"].values, "observed": y_test.values})
    for name, pred in all_test.items():
        predictions[name] = pred
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

    fitted_train = pd.DataFrame({"target_date": train["target_date"].values, "observed": y_train.values})
    for name, pred in all_train.items():
        fitted_train[name] = pred
    fitted_train.to_csv(RESULTS / "train_predictions.csv", index=False)

    report = {
        "design": {
            "task": "one-month-ahead forecast of monthly mean temperature",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_span": f"{train['target_date'].min().date()} .. {train['target_date'].max().date()}",
            "test_span": f"{test['target_date'].min().date()} .. {test['target_date'].max().date()}",
            "predictors": PREDICTORS,
            "cv": "TimeSeriesSplit(n_splits=5) on the training period",
        },
        "tuning": tuning,
        "performance": table,
        "best_model": best,
        "importance": importance_report(fitted, test),
    }

    print(save_json(report, "03_model_performance.json"))


if __name__ == "__main__":
    main()
