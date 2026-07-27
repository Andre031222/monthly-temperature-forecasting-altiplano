import json

import numpy as np
import pandas as pd

from config import DATA, RESULTS, save_json

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


def load():
    audit = json.loads((RESULTS / "01_data_audit.json").read_text(encoding="utf-8"))
    calibration = json.loads((RESULTS / "02_calibration.json").read_text(encoding="utf-8"))
    performance = json.loads((RESULTS / "03_model_performance.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(DATA / "monthly_calibrated.csv", parse_dates=["ym"])
    return audit, calibration, performance, monthly


def table_record(audit, calibration, monthly):
    station = audit["PUNO"]
    series = calibration["monthly_series"]
    valid = monthly.dropna(subset=["tmean"])

    rows = [
        ("Record span", f"{station['start']} to {station['end']}"),
        ("Daily records available", f"{station['rows']:,}"),
        ("Calendar days in span", f"{station['expected_days']:,}"),
        ("Absent calendar days", f"{station['absent_days']:,} ({station['absent_days']/station['expected_days']*100:.1f}%)"),
        ("Missing daily mean temperature", f"{station['missing_pct']['tmean']:.2f}%"),
        ("Missing daily maximum temperature", f"{station['missing_pct']['tmax']:.2f}%"),
        ("Missing daily minimum temperature", f"{station['missing_pct']['tmin']:.2f}%"),
        ("Missing daily relative humidity", f"{station['missing_pct']['rh']:.2f}%"),
        ("Missing daily precipitation", f"{station['missing_pct']['precip']:.2f}%"),
        ("Physically inconsistent temperature records", f"{station['inconsistent_temperature_rows']}"),
        ("Daily values recovered by calibration", f"{calibration['calibration']['days_filled']:,}"),
        ("Candidate months", f"{series['months_total']}"),
        ("Months meeting WMO completeness", f"{series['months_valid']} ({series['pct_valid']:.1f}%)"),
        ("Mean of the monthly series", f"{series['mean_C']:.2f} °C"),
        ("Standard deviation of the monthly series", f"{series['sd_C']:.2f} °C"),
        ("Range of the monthly series", f"{valid['tmean'].min():.2f} to {valid['tmean'].max():.2f} °C"),
        ("Linear trend", f"+{series['trend_C_per_decade']:.2f} ± {series['trend_std_err']:.2f} °C decade⁻¹ (p = {series['trend_p_value']:.4f})"),
    ]
    return pd.DataFrame(rows, columns=["Characteristic", "Value"])


def table_gaps(audit):
    rows = []
    for block in audit["PUNO"]["absent_blocks"]:
        start = pd.Timestamp(block["from"])
        end = pd.Timestamp(block["to"])
        label = start.strftime("%B %Y") if start.to_period("M") == end.to_period("M") else \
            f"{start.strftime('%B %Y')} – {end.strftime('%B %Y')}"
        rows.append({"Period absent from the record": label, "Days": block["days"]})
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {"Period absent from the record": "Total", "Days": frame["Days"].sum()}
    return frame


def table_climatology(calibration, monthly):
    series = calibration["monthly_series"]
    valid = monthly.dropna(subset=["tmean"])
    rows = []
    for index, name in enumerate(MONTH_NAMES, start=1):
        subset = valid[valid["month"] == index]
        rows.append({
            "Month": name,
            "n": len(subset),
            "Mean (°C)": round(series["climatology_C"][str(index)], 2),
            "SD (°C)": round(series["climatology_sd_C"][str(index)], 2),
            "Minimum (°C)": round(float(subset["tmean"].min()), 2),
            "Maximum (°C)": round(float(subset["tmean"].max()), 2),
            "Mean RH (%)": round(float(subset["rh"].mean()), 1),
            "Mean precipitation (mm)": round(float(subset["precip"].mean()), 1),
        })
    return pd.DataFrame(rows)


def table_performance(performance):
    order = ["Persistence", "Climatology", "Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"]
    rows = []
    for name in order:
        entry = performance["performance"][name]
        test = entry["test"]
        dm = entry.get("diebold_mariano_vs_climatology")
        rows.append({
            "Model": name,
            "MAE (°C)": f"{test['MAE']:.3f}",
            "MAE 95% CI": f"[{entry['mae_ci95'][0]:.3f}, {entry['mae_ci95'][1]:.3f}]",
            "RMSE (°C)": f"{test['RMSE']:.3f}",
            "R²": f"{test['R2']:.3f}",
            "MBE (°C)": f"{test['MBE']:+.3f}",
            "MSSS": f"{entry['skill_vs_climatology']:+.3f}",
            "DM p-value": "—" if dm is None else f"{dm['p_value']:.3f}",
        })
    return pd.DataFrame(rows)


def table_generalisation(performance):
    rows = []
    for name, entry in performance["performance"].items():
        if name in ("Climatology", "Persistence"):
            continue
        train_mae = entry["train"]["MAE"]
        test_mae = entry["test"]["MAE"]
        rows.append({
            "Model": name,
            "Training MAE (°C)": f"{train_mae:.3f}",
            "Test MAE (°C)": f"{test_mae:.3f}",
            "Ratio": f"{test_mae / train_mae:.2f}",
            "Cross-validated MAE (°C)": f"{performance['tuning'][name]['cv_mae_C']:.3f}",
            "Selected hyperparameters": ", ".join(
                f"{k.replace('model__', '')} = {v}" for k, v in performance["tuning"][name]["best_params"].items()
            ) or "none (no free hyperparameters)",
        })
    return pd.DataFrame(rows)


def table_importance(performance):
    importance = performance["importance"]
    rf = {item["feature"]: item for item in importance["permutation_importance_random_forest"]}
    mlp = {item["feature"]: item for item in importance["permutation_importance_multilayer_perceptron"]}
    coefficients = importance["linear_coefficients"]

    rows = []
    for feature in sorted(mlp, key=lambda f: mlp[f]["mean"], reverse=True):
        rows.append({
            "Predictor": feature,
            "MLR standardised coefficient": f"{coefficients[feature]:+.3f}",
            "RF permutation importance (°C)": f"{rf[feature]['mean']:+.3f} ± {rf[feature]['sd']:.3f}",
            "MLP permutation importance (°C)": f"{mlp[feature]['mean']:+.3f} ± {mlp[feature]['sd']:.3f}",
        })
    return pd.DataFrame(rows)


def main():
    audit, calibration, performance, monthly = load()

    tables = {
        "table1_record_characteristics": table_record(audit, calibration, monthly),
        "table2_missing_blocks": table_gaps(audit),
        "table3_monthly_climatology": table_climatology(calibration, monthly),
        "table4_model_performance": table_performance(performance),
        "table5_generalisation": table_generalisation(performance),
        "table6_predictor_importance": table_importance(performance),
    }

    summary = {}
    for name, frame in tables.items():
        frame.to_csv(RESULTS / f"{name}.csv", index=False)
        summary[name] = {"rows": len(frame), "columns": list(frame.columns)}
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))

    save_json(summary, "05_tables_index.json")


if __name__ == "__main__":
    main()
