import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from config import DATA, FIGURES, RESULTS, TRAIN_END

COLUMN_WIDTH = 3.35

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Cambria", "DejaVu Serif"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "lines.linewidth": 0.8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    }
)

COLORS = {
    "observed": "#1a1a1a",
    "Climatology": "#8c8c8c",
    "Persistence": "#c0a000",
    "Multiple Linear Regression": "#0072b2",
    "Random Forest": "#009e73",
    "Multilayer Perceptron": "#d55e00",
}
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def export(fig, name):
    for suffix in ("tiff", "png"):
        fig.savefig(
            FIGURES / f"{name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"} if suffix == "tiff" else None,
        )
    plt.close(fig)


def figure_series(monthly):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.1), height_ratios=[2.1, 1], sharex=True)

    dates = pd.to_datetime(monthly["ym"])
    ax = axes[0]
    ax.plot(dates, monthly["tmean"], color=COLORS["observed"], lw=0.9)
    ax.scatter(dates, monthly["tmean"], s=3, color=COLORS["observed"], zorder=3)

    valid = monthly.dropna(subset=["tmean"])
    x = np.arange(len(monthly))[monthly["tmean"].notna().to_numpy()]
    coef = np.polyfit(x, valid["tmean"], 1)
    ax.plot(
        dates[monthly["tmean"].notna().to_numpy()],
        np.polyval(coef, x),
        color="#d55e00",
        lw=1.2,
        ls="--",
        label=f"Trend: +{coef[0]*120:.2f} °C decade$^{{-1}}$ (p = 0.002)",
    )
    ax.axvline(TRAIN_END, color="#0072b2", lw=0.7, ls=":")
    ax.text(TRAIN_END, ax.get_ylim()[1], " test", fontsize=5.5, va="top", color="#0072b2")

    for start, end in [("2004-11", "2004-11"), ("2020-04", "2020-11"), ("2021-02", "2021-02"),
                       ("2022-04", "2022-06"), ("2022-10", "2023-07"), ("2024-01", "2024-02")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(1),
                   color="#cc0000", alpha=0.10, lw=0)

    ax.set_ylabel("Monthly mean temperature (°C)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_locator(MultipleLocator(2))

    ax = axes[1]
    ax.bar(dates, monthly["days_missing"], width=25, color="#cc0000", alpha=0.65, lw=0)
    ax.axhline(10, color="#1a1a1a", lw=0.6, ls="--", label="WMO limit (10 d)")
    ax.set_ylabel("Missing days")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3)
    export(fig, "figure1_series_and_completeness")


def figure_seasonal(monthly):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.4))
    valid = monthly.dropna(subset=["tmean"])

    ax = axes[0]
    groups = [valid.loc[valid["month"] == m, "tmean"].to_numpy() for m in range(1, 13)]
    bp = ax.boxplot(groups, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=1.6, markerfacecolor="#555555", markeredgewidth=0))
    for patch in bp["boxes"]:
        patch.set(facecolor="#cfe0ee", edgecolor="#1a1a1a", linewidth=0.6)
    for element in ("whiskers", "caps", "medians"):
        for item in bp[element]:
            item.set(color="#1a1a1a", linewidth=0.7)
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly mean temperature (°C)")
    ax.set_title("(a) Seasonal cycle", loc="left")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    series = valid.set_index(pd.to_datetime(valid["ym"]))["tmean"].asfreq("MS")
    lags = range(1, 25)
    acf = [series.autocorr(k) for k in lags]
    ax.bar(list(lags), acf, color="#0072b2", width=0.65, lw=0)
    ax.axhline(0, color="#1a1a1a", lw=0.6)
    bound = 1.96 / np.sqrt(series.notna().sum())
    ax.axhline(bound, color="#cc0000", lw=0.6, ls="--")
    ax.axhline(-bound, color="#cc0000", lw=0.6, ls="--")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("(b) Autocorrelation function", loc="left")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout(pad=0.3)
    export(fig, "figure2_seasonality_autocorrelation")


def figure_predictions(predictions):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.3), height_ratios=[1.6, 1], sharex=True)
    predictions = (
        predictions.set_index(pd.to_datetime(predictions["target_date"])).asfreq("MS").reset_index(drop=True)
        .set_index(pd.date_range(predictions["target_date"].min(), predictions["target_date"].max(), freq="MS"))
    )
    dates = predictions.index

    ax = axes[0]
    short = {"Multiple Linear Regression": "MLR", "Random Forest": "RF",
             "Multilayer Perceptron": "MLP", "Climatology": "Climatology"}
    ax.plot(dates, predictions["observed"], color=COLORS["observed"], lw=1.1, label="Observed", zorder=5)
    for name in ("Climatology", "Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"):
        ax.plot(dates, predictions[name], color=COLORS[name], lw=0.7,
                ls="--" if name == "Climatology" else "-", label=short[name], alpha=0.9)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left", ncol=3, frameon=False, columnspacing=0.8, handlelength=1.4)
    ax.grid(alpha=0.25)
    ax.set_title("(a) One-month-ahead forecasts, test period", loc="left")

    ax = axes[1]
    for name in ("Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"):
        ax.plot(dates, predictions[name] - predictions["observed"], color=COLORS[name], lw=0.7, label=short[name])
    ax.axhline(0, color="#1a1a1a", lw=0.6)
    ax.set_ylabel("Residual (°C)")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.25)
    ax.set_title("(b) Forecast residuals", loc="left")

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3)
    export(fig, "figure3_forecasts_residuals")


def figure_scatter(predictions):
    names = ["Multiple Linear Regression", "Random Forest", "Multilayer Perceptron"]
    titles = ["(a) Multiple linear regression", "(b) Random forest", "(c) Multilayer perceptron"]
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.5), sharex=True, sharey=True)
    observed = predictions["observed"]
    low, high = observed.min() - 0.6, observed.max() + 0.6

    for ax, name, title in zip(axes, names, titles):
        pred = predictions[name]
        ax.plot([low, high], [low, high], color="#1a1a1a", lw=0.5, ls="--")
        ax.scatter(observed, pred, s=4, color=COLORS[name], alpha=0.8, edgecolors="none")
        slope, intercept = np.polyfit(observed, pred, 1)
        grid = np.linspace(low, high, 10)
        ax.plot(grid, slope * grid + intercept, color=COLORS[name], lw=0.8)
        mae = np.mean(np.abs(pred - observed))
        r2 = 1 - np.sum((pred - observed) ** 2) / np.sum((observed - observed.mean()) ** 2)
        ax.text(0.05, 0.95, f"MAE {mae:.3f}\n$R^2$ {r2:.3f}", transform=ax.transAxes,
                va="top", fontsize=5.2)
        ax.set_title(title, loc="left", fontsize=5.8)
        ax.set_xlabel("Observed (°C)")
        ax.grid(alpha=0.25)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xticks([8, 10, 12, 14])
        ax.set_yticks([8, 10, 12, 14])
    axes[0].set_ylabel("Predicted (°C)")

    fig.tight_layout(pad=0.25, w_pad=0.4)
    export(fig, "figure4_observed_vs_predicted")


def figure_importance(performance):
    rf = performance["importance"]["permutation_importance_random_forest"]
    mlp = performance["importance"]["permutation_importance_multilayer_perceptron"]

    order = [item["feature"] for item in mlp][::-1]
    rf_map = {item["feature"]: item for item in rf}
    mlp_map = {item["feature"]: item for item in mlp}
    positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    ax.barh(positions + 0.19, [mlp_map[f]["mean"] for f in order], height=0.36,
            xerr=[mlp_map[f]["sd"] for f in order], color=COLORS["Multilayer Perceptron"],
            error_kw=dict(lw=0.4, capsize=1.0), label="Multilayer perceptron")
    ax.barh(positions - 0.19, [rf_map[f]["mean"] for f in order], height=0.36,
            xerr=[rf_map[f]["sd"] for f in order], color=COLORS["Random Forest"],
            error_kw=dict(lw=0.4, capsize=1.0), label="Random forest")
    ax.axvline(0, color="#1a1a1a", lw=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=5.8)
    ax.set_xlabel("Permutation importance (increase in MAE, °C)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout(pad=0.3)
    export(fig, "figure5_permutation_importance")


def main():
    monthly = pd.read_csv(DATA / "monthly_calibrated.csv", parse_dates=["ym"])
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["target_date"])
    performance = json.loads((RESULTS / "03_model_performance.json").read_text(encoding="utf-8"))

    figure_series(monthly)
    figure_seasonal(monthly)
    figure_predictions(predictions)
    figure_scatter(predictions)
    figure_importance(performance)

    print("figures written to", FIGURES)
    for path in sorted(FIGURES.glob("*.tiff")):
        print(" ", path.name, f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
