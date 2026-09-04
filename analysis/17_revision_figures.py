"""Build the three consolidated figures reported in the paper.

The journal limits tables and figures to seven items combined, so the granular
figures written by scripts 04, 07 and 14 are merged into three multi-panel
figures that keep every panel a reader needs to follow the argument. All panels
are redrawn from the stored predictions and JSON reports, so no model is
refitted here.
"""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.ticker import MultipleLocator
from scipy import stats

from config import DATA, FIGURES, RESULTS, TRAIN_END, save_json

COLUMN_WIDTH = 3.35

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "DejaVu Serif"],
    "font.size": 6.5,
    "axes.labelsize": 6.3,
    "axes.titlesize": 6.6,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

COLORS = {
    "observed": "#1a1a1a",
    "Climatology": "#8c8c8c",
    "Persistence": "#c0a000",
    "Multiple Linear Regression": "#0072b2",
    "Random Forest": "#009e73",
    "Multilayer Perceptron": "#d55e00",
}
SHORT = {
    "Multiple Linear Regression": "MLR",
    "Random Forest": "RF",
    "Multilayer Perceptron": "MLP",
}
MODELS = list(SHORT)
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
ABSENT_BLOCKS = [("2004-11", "2004-11"), ("2020-04", "2020-11"), ("2021-02", "2021-02"),
                 ("2022-04", "2022-06"), ("2022-10", "2023-07"), ("2024-01", "2024-02")]


def export(fig, name):
    for suffix in ("tiff", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"} if suffix == "tiff" else None)
    plt.close(fig)


def figure1_record(monthly):
    """(a) series and absent blocks, (b) missing days, (c) seasonal cycle, (d) ACF."""
    fig = plt.figure(figsize=(COLUMN_WIDTH, 4.75))
    grid = fig.add_gridspec(4, 1, height_ratios=[2.0, 0.85, 1.35, 1.15], hspace=0.85)
    ax_series = fig.add_subplot(grid[0])
    ax_gaps = fig.add_subplot(grid[1], sharex=ax_series)
    ax_cycle = fig.add_subplot(grid[2])
    ax_acf = fig.add_subplot(grid[3])

    dates = pd.to_datetime(monthly["ym"])
    observed = monthly["tmean"].notna().to_numpy()

    ax = ax_series
    ax.plot(dates, monthly["tmean"], color=COLORS["observed"], lw=0.9)
    ax.scatter(dates, monthly["tmean"], s=2.5, color=COLORS["observed"], zorder=3)
    index = np.arange(len(monthly))[observed]
    coefficients = np.polyfit(index, monthly["tmean"].dropna(), 1)
    ax.plot(dates[observed], np.polyval(coefficients, index), color="#d55e00", lw=1.1, ls="--",
            label=f"Trend: +{coefficients[0] * 120:.2f} °C decade$^{{-1}}$ (p = 0.002)")
    ax.axvline(TRAIN_END, color="#0072b2", lw=0.7, ls=":")
    ax.text(TRAIN_END, ax.get_ylim()[1], " test", fontsize=5.2, va="top", color="#0072b2")
    for start, end in ABSENT_BLOCKS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(1),
                   color="#cc0000", alpha=0.10, lw=0)
    ax.set_ylabel("Monthly mean\ntemperature (°C)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_title("(a) Retained monthly series and absent blocks", loc="left")
    ax.tick_params(labelbottom=False)

    ax = ax_gaps
    ax.bar(dates, monthly["days_missing"], width=25, color="#cc0000", alpha=0.65, lw=0)
    ax.axhline(10, color="#1a1a1a", lw=0.6, ls="--", label="WMO limit (10 d)")
    ax.set_ylabel("Missing\ndays")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    ax.set_title("(b) Daily values absent from each month", loc="left")

    ax = ax_cycle
    valid = monthly.dropna(subset=["tmean"])
    groups = [valid.loc[valid["month"] == m, "tmean"].to_numpy() for m in range(1, 13)]
    box = ax.boxplot(groups, widths=0.6, patch_artist=True, showfliers=True,
                     flierprops=dict(marker="o", markersize=1.4, markerfacecolor="#555555",
                                     markeredgewidth=0))
    for patch in box["boxes"]:
        patch.set(facecolor="#cfe0ee", edgecolor="#1a1a1a", linewidth=0.6)
    for element in ("whiskers", "caps", "medians"):
        for item in box[element]:
            item.set(color="#1a1a1a", linewidth=0.7)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(c) Seasonal cycle by calendar month", loc="left")

    ax = ax_acf
    series = valid.set_index(pd.to_datetime(valid["ym"]))["tmean"].asfreq("MS")
    lags = list(range(1, 25))
    ax.bar(lags, [series.autocorr(k) for k in lags], color="#0072b2", width=0.65, lw=0)
    bound = 1.96 / np.sqrt(series.notna().sum())
    ax.axhline(0, color="#1a1a1a", lw=0.6)
    ax.axhline(bound, color="#cc0000", lw=0.6, ls="--")
    ax.axhline(-bound, color="#cc0000", lw=0.6, ls="--")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(d) Autocorrelation function", loc="left")

    export(fig, "figure1_record_and_structure")


def figure2_performance(predictions, extended):
    """(a) forecasts, (b) residuals, (c) observed vs predicted, (d) QQ, (e) seasonal MAE."""
    fig = plt.figure(figsize=(COLUMN_WIDTH, 5.1))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.7, 0.95, 1.25, 1.25], hspace=0.95, wspace=0.4)
    ax_forecast = fig.add_subplot(grid[0, :])
    ax_residual = fig.add_subplot(grid[1, :], sharex=ax_forecast)
    scatter_axes = [fig.add_subplot(grid[2, i]) for i in range(3)]
    qq_axes = [fig.add_subplot(grid[3, i]) for i in range(2)]
    ax_season = fig.add_subplot(grid[3, 2])

    frame = predictions.set_index(pd.to_datetime(predictions["target_date"]))
    # Reindex onto a complete monthly grid so the absent months appear as breaks
    # in the lines rather than as straight segments spanning a two-year gap.
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="MS"))
    dates = frame.index

    ax = ax_forecast
    ax.plot(dates, frame["observed"], color=COLORS["observed"], lw=1.0, label="Observed", zorder=5)
    for name in ("Climatology", *MODELS):
        ax.plot(dates, frame[name], color=COLORS[name], lw=0.65,
                ls="--" if name == "Climatology" else "-",
                label=SHORT.get(name, name), alpha=0.9)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False,
              columnspacing=0.8, handlelength=1.2, borderpad=0.1)
    ax.grid(alpha=0.25)
    ax.set_title("(a) One-month-ahead forecasts over the test period", loc="left", pad=14)
    ax.tick_params(labelbottom=False)

    ax = ax_residual
    for name in MODELS:
        ax.plot(dates, frame[name] - frame["observed"], color=COLORS[name], lw=0.65, label=SHORT[name])
    ax.axhline(0, color="#1a1a1a", lw=0.6)
    ax.set_ylabel("Residual (°C)")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.25)
    ax.set_title("(b) Forecast residuals", loc="left")

    complete = frame.dropna(subset=["observed"])
    observed = complete["observed"]
    low, high = observed.min() - 0.6, observed.max() + 0.6
    titles = ["(c) MLR", "(d) RF", "(e) MLP"]
    for ax, name, title in zip(scatter_axes, MODELS, titles):
        prediction = complete[name]
        ax.plot([low, high], [low, high], color="#1a1a1a", lw=0.5, ls="--")
        ax.scatter(observed, prediction, s=3, color=COLORS[name], alpha=0.8, edgecolors="none")
        slope, intercept = np.polyfit(observed, prediction, 1)
        span = np.linspace(low, high, 10)
        ax.plot(span, slope * span + intercept, color=COLORS[name], lw=0.7)
        mae = float(np.mean(np.abs(prediction - observed)))
        ax.text(0.05, 0.95, f"MAE {mae:.3f}", transform=ax.transAxes, va="top", fontsize=5.0)
        ax.set_title(title, loc="left", fontsize=5.8)
        ax.set_xlabel("Observed (°C)")
        ax.grid(alpha=0.25)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xticks([8, 10, 12, 14])
        ax.set_yticks([8, 10, 12, 14])
    scatter_axes[0].set_ylabel("Predicted (°C)")

    residuals = extended["residuals"]
    for ax, name, letter in zip(qq_axes, ("Multiple Linear Regression", "Random Forest"), "fg"):
        values = (complete[name] - observed).to_numpy()
        stats.probplot(values, dist="norm", plot=None)
        theoretical = stats.probplot(values, dist="norm", fit=False)[0]
        ax.scatter(theoretical, np.sort(values), s=3, color=COLORS[name], edgecolors="none")
        line = np.linspace(theoretical.min(), theoretical.max(), 10)
        ax.plot(line, line * values.std(ddof=1) + values.mean(), color="#1a1a1a", lw=0.5, ls="--")
        ax.set_title(f"({letter}) {SHORT[name]} Q-Q", loc="left", fontsize=5.8)
        ax.set_xlabel("Theoretical")
        ax.grid(alpha=0.25)
        ax.text(0.05, 0.95, f"W p {residuals[name]['shapiro_p']:.3f}",
                transform=ax.transAxes, va="top", fontsize=5.0)
    qq_axes[0].set_ylabel("Residual (°C)")

    ax = ax_season
    seasons = list(extended["seasonal"])
    positions = np.arange(len(seasons))
    width = 0.26
    for offset, name in zip((-width, 0, width), MODELS):
        ax.bar(positions + offset, [extended["seasonal"][s][name]["MAE"] for s in seasons],
               width=width, color=COLORS[name], label=SHORT[name])
    ax.axhline(0.597, color="#1a1a1a", lw=0.6, ls="--")
    ax.text(0.03, 0.90, "climatology", transform=ax.transAxes, fontsize=4.6, color="#1a1a1a")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Wet", "Dry", "Trans."], fontsize=5.2)
    ax.set_ylabel("MAE (°C)")
    ax.set_ylim(0, 0.78)
    ax.set_title("(h) MAE by season", loc="left", fontsize=5.8)
    ax.grid(alpha=0.25, axis="y")

    export(fig, "figure2_forecast_performance")


def figure3_replication(replication, monthly_validation):
    """(a) MAE by site, (b) pooled learning curve, (c) reanalysis against station data."""
    fig = plt.figure(figsize=(COLUMN_WIDTH, 4.35))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.15, 1.15], hspace=0.95, wspace=0.45)
    ax_sites = fig.add_subplot(grid[0, :])
    ax_curve = fig.add_subplot(grid[1, :])
    ax_series = fig.add_subplot(grid[2, 0])
    ax_scatter = fig.add_subplot(grid[2, 1])

    per_station = replication["per_station"]
    stations = sorted(per_station, key=lambda s: per_station[s]["Multiple Linear Regression"]["MAE"])
    positions = np.arange(len(stations))

    ax = ax_sites
    width = 0.26
    for offset, name in zip((-width, 0, width), MODELS):
        ax.bar(positions + offset, [per_station[s][name]["MAE"] for s in stations],
               width=width, color=COLORS[name], label=SHORT[name])
    ax.plot(positions, [per_station[s]["Climatology"]["MAE"] for s in stations],
            color="#1a1a1a", lw=0.7, ls="--", marker="o", ms=1.8, label="Climatology")
    ax.set_xticks(positions)
    ax.set_xticklabels([s.replace("_", " ") for s in stations], rotation=40,
                       ha="right", fontsize=4.6)
    ax.set_ylabel("Test MAE (°C)")
    ax.set_ylim(0, 0.92)
    ax.legend(loc="upper left", ncol=4, frameon=False, columnspacing=0.7, handlelength=1.1)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("(a) Replication across eleven independent series", loc="left")

    ax = ax_curve
    for name in MODELS:
        curve = replication["pooled_learning_curve"][name]
        sizes = [point["n_train"] for point in curve]
        values = [point["MAE"] for point in curve]
        errors = [point["sd"] for point in curve]
        ax.plot(sizes, values, color=COLORS[name], marker="o", ms=2.2, label=SHORT[name])
        ax.fill_between(sizes, np.array(values) - np.array(errors),
                        np.array(values) + np.array(errors), color=COLORS[name], alpha=0.15, lw=0)
    ax.axvline(137, color="#8c8c8c", lw=0.6, ls=":")
    ax.text(137, ax.get_ylim()[1], " Puno record", fontsize=5.0, va="top", color="#5c5c5c")
    ax.set_xscale("log")
    ax.set_ylim(0.39, 0.68)
    ax.set_xlabel("Pooled training months (log scale)")
    ax.set_ylabel("Test MAE (°C)")
    ax.legend(loc="upper right", frameon=False, ncol=3, columnspacing=0.7, handlelength=1.1)
    ax.grid(alpha=0.25)
    ax.set_title("(b) Learning curves to 2,475 pooled training months", loc="left")

    dates = pd.to_datetime(monthly_validation["ym"])
    ax = ax_series
    ax.plot(dates, monthly_validation["tmean_senamhi"], color="#1a1a1a", lw=0.7, label="SENAMHI")
    ax.plot(dates, monthly_validation["tmean_nasa"], color="#0072b2", lw=0.7, label="NASA")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Year")
    ax.set_ylim(3.5, 18.5)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=4.8,
              columnspacing=0.6, handlelength=1.0, borderpad=0.1)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
    ax.grid(alpha=0.25)
    ax.set_title("(c) Reanalysis vs station", loc="left", fontsize=5.8)

    ax = ax_scatter
    senamhi = monthly_validation["tmean_senamhi"]
    nasa = monthly_validation["tmean_nasa"]
    ax.scatter(senamhi, nasa, s=3, color="#0072b2", alpha=0.75, edgecolors="none")
    slope, intercept = np.polyfit(senamhi, nasa, 1)
    span = np.linspace(senamhi.min() - 0.3, senamhi.max() + 0.3, 10)
    ax.plot(span, slope * span + intercept, color="#1a1a1a", lw=0.6)
    # Reported alongside the raw correlation, the deseasonalised correlation shows
    # how much of the apparent agreement is carried by the shared annual cycle.
    month = monthly_validation["ym"].dt.month
    senamhi_anomaly = senamhi - senamhi.groupby(month).transform("mean")
    nasa_anomaly = nasa - nasa.groupby(month).transform("mean")
    correlation = float(senamhi.corr(nasa))
    anomaly_correlation = float(senamhi_anomaly.corr(nasa_anomaly))
    bias = float((nasa - senamhi).mean())
    ax.text(0.05, 0.95,
            f"r = {correlation:.3f}\nr' = {anomaly_correlation:.3f}\nbias {bias:+.2f} °C",
            transform=ax.transAxes, va="top", fontsize=5.0)
    ax.set_xlabel("SENAMHI (°C)")
    ax.set_ylabel("NASA POWER (°C)")
    ax.grid(alpha=0.25)
    ax.set_title("(d) Monthly agreement", loc="left", fontsize=5.8)

    export(fig, "figure3_replication_and_scaling")
    return {
        "n_months": int(len(monthly_validation)),
        "pearson_r": round(correlation, 4),
        "deseasonalised_anomaly_r": round(anomaly_correlation, 4),
        "bias_C": round(bias, 3),
    }


def main():
    monthly = pd.read_csv(DATA / "monthly_calibrated.csv", parse_dates=["ym"])
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["target_date"])
    extended = json.loads((RESULTS / "06_extended_analysis.json").read_text(encoding="utf-8"))
    replication = json.loads((RESULTS / "13_multistation_replication.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(RESULTS / "nasa_vs_senamhi_monthly.csv", parse_dates=["ym"])

    figure1_record(monthly)
    figure2_performance(predictions, extended)
    agreement = figure3_replication(replication, validation)
    print(save_json({"reanalysis_agreement": agreement}, "17_revision_figures.json"))

    for name in ("figure1_record_and_structure", "figure2_forecast_performance",
                 "figure3_replication_and_scaling"):
        path = FIGURES / f"{name}.tiff"
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
