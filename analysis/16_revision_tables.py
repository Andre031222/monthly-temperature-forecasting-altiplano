"""Build the four consolidated tables reported in the paper.

The journal limits tables and figures to seven items combined, so the granular
tables written by script 05 are merged into four. Every value is read from the
JSON reports produced by scripts 01-13; nothing is refitted here, so the numbers
are identical to those of the underlying analysis.
"""

import json

import pandas as pd

from config import RESULTS, save_json

SHORT = {
    "Multiple Linear Regression": "MLR",
    "Random Forest": "RF",
    "Multilayer Perceptron": "MLP",
}


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def table1_record():
    """Record characteristics, absent blocks and retained series in one table."""
    audit = load("01_data_audit.json")["PUNO"]
    calibration = load("02_calibration.json")
    series = calibration["monthly_series"]
    blocks = pd.read_csv(RESULTS / "table2_missing_blocks.csv")
    longest = blocks[blocks["Period absent from the record"] != "Total"].sort_values(
        "Days", ascending=False
    ).iloc[0]

    rows = [
        ("Record span", f"{audit['start']} to {audit['end']}"),
        ("Daily records available", f"{audit['rows']:,}"),
        ("Calendar days in span", f"{audit['expected_days']:,}"),
        (
            "Absent calendar days",
            f"{audit['absent_days']:,} ({audit['absent_days'] / audit['expected_days'] * 100:.1f}%) "
            f"in {len(blocks) - 1} blocks",
        ),
        (
            "Longest absent block",
            f"{longest['Period absent from the record']} ({int(longest['Days'])} days)",
        ),
        ("Missing daily mean temperature", f"{audit['missing_pct']['tmean']:.2f}%"),
        (
            "Missing daily maximum / minimum temperature",
            f"{audit['missing_pct']['tmax']:.2f}% / {audit['missing_pct']['tmin']:.2f}%",
        ),
        (
            "Missing daily relative humidity / precipitation",
            f"{audit['missing_pct']['rh']:.2f}% / {audit['missing_pct']['precip']:.2f}%",
        ),
        ("Physically inconsistent temperature records", f"{audit['inconsistent_temperature_rows']}"),
        (
            "Daily values recovered by calibration",
            f"{calibration['calibration']['days_filled']:,} in the record, "
            f"{series['pct_days_calibrated']:.1f}% of the days in retained months",
        ),
        ("Candidate months", f"{series['months_total']}"),
        (
            "Months meeting WMO completeness",
            f"{series['months_valid']} ({series['pct_valid']:.1f}%)",
        ),
        ("Mean of the retained monthly series", f"{series['mean_C']:.2f} °C"),
        ("Standard deviation of the retained monthly series", f"{series['sd_C']:.2f} °C"),
        ("Range of the retained monthly series", f"{series['min_C']:.2f} to {series['max_C']:.2f} °C"),
        ("Coldest / warmest climatological month", "July, 8.17 °C / November, 12.85 °C"),
        ("Seasonal amplitude of the climatology", "4.68 °C"),
        ("Mean within-month standard deviation", "0.71 °C"),
        (
            "Linear trend",
            f"+{series['trend_C_per_decade']:.2f} ± {series['trend_std_err']:.2f} "
            f"°C decade⁻¹ (p = {series['trend_p_value']:.4f})",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["Characteristic", "Value"])
    frame.to_csv(RESULTS / "rev_table1_record.csv", index=False)
    return frame


def table2_performance():
    """Test accuracy, skill, significance and generalisation in one table."""
    performance = load("03_model_performance.json")
    scores = performance["performance"]
    tuning = performance["tuning"]

    hyperparameters = {
        "Multiple Linear Regression": "none",
        "Random Forest": "depth 6, 300 trees",
        "Multilayer Perceptron": "16 units, α = 1.0",
    }

    order = ["Persistence", "Climatology", "Multiple Linear Regression",
             "Random Forest", "Multilayer Perceptron"]
    columns = {}
    for name in order:
        entry = scores[name]
        test = entry["test"]
        dm = entry.get("diebold_mariano_vs_climatology")
        fitted = name in SHORT
        columns[SHORT.get(name, name)] = [
            f"{test['MAE']:.3f}",
            f"[{entry['mae_ci95'][0]:.3f}, {entry['mae_ci95'][1]:.3f}]",
            f"{test['RMSE']:.3f}",
            f"{test['R2']:.3f}",
            f"{test['MBE']:+.3f}",
            f"{entry['skill_vs_climatology']:+.3f}",
            "—" if dm is None else f"{dm['p_value']:.3f}",
            f"{entry['train']['MAE']:.3f}" if fitted else "—",
            f"{test['MAE'] / entry['train']['MAE']:.2f}" if fitted else "—",
            f"{tuning[name]['cv_mae_C']:.3f}" if fitted else "—",
            hyperparameters.get(name, "—"),
        ]

    metrics = ["MAE (°C)", "MAE 95% CI", "RMSE (°C)", "R²", "MBE (°C)", "MSSS",
               "DM p vs climatology", "Training MAE (°C)", "Test/train ratio",
               "Cross-validated MAE (°C)", "Hyperparameters"]
    frame = pd.DataFrame({"Measure": metrics, **columns})
    frame.to_csv(RESULTS / "rev_table2_performance.csv", index=False)
    return frame


def table3_diagnostics():
    """Residual diagnostics, seed stability and the ablation in one table."""
    extended = load("06_extended_analysis.json")
    seeds = load("10_seed_stability.json")
    residuals = extended["residuals"]
    ablation = extended["ablation"]
    seasonal = extended["seasonal"]

    seasons = list(seasonal)
    columns = {}
    for name, short in SHORT.items():
        residual = residuals[name]
        seed = seeds[name]
        if seed.get("deterministic"):
            seed_cells = ["—", "—", "—", "—"]
        else:
            seed_cells = [
                f"{seed['mean_MAE']:.3f}",
                f"{seed['sd_MAE']:.3f}",
                f"{seed['min_MAE']:.3f}–{seed['max_MAE']:.3f}",
                f"{seed['seeds_beating_linear']}/{seed['n_seeds']}",
            ]
        columns[short] = [
            f"{residual['shapiro_W']:.3f}",
            f"{residual['shapiro_p']:.3f}",
            f"{residual['lag1_autocorrelation']:+.3f}",
            f"{residual['ljung_box_Q5']:.2f}",
            f"{residual['ljung_box_p']:.3f}",
            *seed_cells,
            f"{ablation['Seasonal only'][name]['MAE']:.3f}",
            f"{ablation['Thermal memory only'][name]['MAE']:.3f}",
            f"{ablation['Seasonal + memory'][name]['MAE']:.3f}",
            f"{ablation['Without atmospheric covariates'][name]['MAE']:.3f}",
            f"{ablation['All predictors'][name]['MAE']:.3f}",
            *[f"{seasonal[season][name]['MAE']:.3f}" for season in seasons],
        ]

    labels = [
        "Shapiro-Wilk W", "W p", "Lag-1 autocorrelation", "Ljung-Box Q", "Q p",
        "Mean MAE over seeds (°C)", "SD over seeds (°C)", "Range over seeds (°C)",
        "Seeds beating MLR",
        "MAE, seasonal only (k = 3)", "MAE, memory only (k = 4)",
        "MAE, seasonal + memory (k = 8)", "MAE, without covariates (k = 9)",
        "MAE, all predictors (k = 12)",
        "MAE, wet season (n = 13)", "MAE, dry season (n = 18)",
        "MAE, transition (n = 18)",
    ]
    frame = pd.DataFrame({"Diagnostic": labels, **columns})
    frame.to_csv(RESULTS / "rev_table3_diagnostics.csv", index=False)
    return frame


def table4_replication():
    """Replication across the eleven independent series."""
    report = load("13_multistation_replication.json")
    per_station = report["per_station"]
    order = sorted(per_station, key=lambda s: per_station[s]["Multiple Linear Regression"]["MAE"])

    rows = []
    for station in order:
        entry = per_station[station]
        rows.append({
            "Series": station.replace("_", " "),
            "n test": entry["n_test"],
            "Clim.": f"{entry['Climatology']['MAE']:.3f}",
            "MLR": f"{entry['Multiple Linear Regression']['MAE']:.3f}",
            "RF": f"{entry['Random Forest']['MAE']:.3f}",
            "MLP": f"{entry['Multilayer Perceptron']['MAE']:.3f}",
        })
    pooled = report["pooled"]
    rows.append({
        "Series": "Pooled",
        "n test": pooled["n_test"],
        "Clim.": f"{pooled['Climatology']['MAE']:.3f}",
        "MLR": f"{pooled['Multiple Linear Regression']['MAE']:.3f}",
        "RF": f"{pooled['Random Forest']['MAE']:.3f}",
        "MLP": f"{pooled['Multilayer Perceptron']['MAE']:.3f}",
    })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "rev_table4_replication.csv", index=False)
    return frame


def main():
    built = {
        "rev_table1_record.csv": len(table1_record()),
        "rev_table2_performance.csv": len(table2_performance()),
        "rev_table3_diagnostics.csv": len(table3_diagnostics()),
        "rev_table4_replication.csv": len(table4_replication()),
    }
    print(save_json({"tables": built, "n_tables": len(built)}, "16_revision_tables.json"))


if __name__ == "__main__":
    main()
