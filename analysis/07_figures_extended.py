import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import DATA, FIGURES, RESULTS, TRAIN_END

COLUMN_WIDTH = 3.35

plt.rcParams.update({
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
    "lines.linewidth": 0.9,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

COLORS = {
    "Multiple Linear Regression": "#0072b2",
    "Random Forest": "#009e73",
    "Multilayer Perceptron": "#d55e00",
}
SHORT = {
    "Multiple Linear Regression": "MLR",
    "Random Forest": "RF",
    "Multilayer Perceptron": "MLP",
}


def export(fig, name):
    for suffix in ("tiff", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"} if suffix == "tiff" else None)
    plt.close(fig)


def figure_learning_and_season(report, climatology_mae):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.5))

    ax = axes[0]
    sizes = report["learning_curve"]["sizes"]
    for name, points in report["learning_curve"]["curves"].items():
        ax.plot(sizes, [p["MAE"] for p in points], marker="o", markersize=2.4,
                color=COLORS[name], label=SHORT[name])
    ax.axhline(climatology_mae, color="#8c8c8c", ls="--", lw=0.8, label="Climatology")
    ax.set_xlabel("Training months")
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(a) Learning curves", loc="left")
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1.4)
    ax.grid(alpha=0.25)

    ax = axes[1]
    seasons = list(report["seasonal"].keys())
    labels = ["Wet\n(Dec-Mar)", "Dry\n(May-Aug)", "Transition\n(Apr, Sep-Nov)"]
    positions = np.arange(len(seasons))
    width = 0.26
    for offset, name in zip((-width, 0, width), COLORS):
        values = [report["seasonal"][s][name]["MAE"] for s in seasons]
        ax.bar(positions + offset, values, width=width, color=COLORS[name], label=SHORT[name])
    ax.axhline(climatology_mae, color="#8c8c8c", ls="--", lw=0.8, label="Climatology")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (°C)")
    ax.set_title("(b) Error by season", loc="left")
    ax.legend(frameon=False, ncol=4, columnspacing=0.6, handlelength=1.1, loc="upper center")
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(0.75, ax.get_ylim()[1] * 1.28))

    fig.tight_layout(pad=0.3)
    export(fig, "figure6_learning_curve_and_season")


def figure_residual_diagnostics(predictions):
    fig, axes = plt.subplots(2, 3, figsize=(COLUMN_WIDTH, 2.9), sharex="row")
    names = list(COLORS)
    titles = ["(a) MLR", "(b) RF", "(c) MLP"]

    for index, (name, title) in enumerate(zip(names, titles)):
        residuals = (predictions[name] - predictions["observed"]).to_numpy()

        ax = axes[0, index]
        stats.probplot(residuals, dist="norm", plot=None)
        osm, osr = stats.probplot(residuals, dist="norm", fit=False)
        ax.scatter(osm, osr, s=3, color=COLORS[name], edgecolors="none")
        limits = [min(osm.min(), osr.min()) - 0.1, max(osm.max(), osr.max()) + 0.1]
        ax.plot(limits, limits, color="#1a1a1a", lw=0.5, ls="--")
        ax.set_title(title, loc="left", fontsize=6)
        ax.grid(alpha=0.25)
        if index == 0:
            ax.set_ylabel("Residual (°C)")
        ax.set_xlabel("Theoretical quantile")

        ax = axes[1, index]
        lags = range(1, 13)
        series = pd.Series(residuals)
        acf = [series.autocorr(k) for k in lags]
        ax.bar(list(lags), acf, color=COLORS[name], width=0.7, lw=0)
        bound = 1.96 / np.sqrt(len(residuals))
        ax.axhline(bound, color="#cc0000", lw=0.5, ls="--")
        ax.axhline(-bound, color="#cc0000", lw=0.5, ls="--")
        ax.axhline(0, color="#1a1a1a", lw=0.5)
        ax.set_ylim(-0.55, 0.55)
        ax.set_xlabel("Lag (months)")
        if index == 0:
            ax.set_ylabel("Residual ACF")
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout(pad=0.25, w_pad=0.5, h_pad=0.6)
    export(fig, "figure7_residual_diagnostics")


def main():
    report = json.loads((RESULTS / "06_extended_analysis.json").read_text(encoding="utf-8"))
    performance = json.loads((RESULTS / "03_model_performance.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["target_date"])

    climatology_mae = performance["performance"]["Climatology"]["test"]["MAE"]
    figure_learning_and_season(report, climatology_mae)
    figure_residual_diagnostics(predictions)

    for path in sorted(FIGURES.glob("figure[67]*.tiff")):
        print(" ", path.name, f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
