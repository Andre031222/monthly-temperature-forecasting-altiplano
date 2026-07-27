import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES, RESULTS

COLUMN_WIDTH = 3.35

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "DejaVu Serif"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7,
    "xtick.labelsize": 5.8,
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
SHORT = {"Multiple Linear Regression": "MLR", "Random Forest": "RF", "Multilayer Perceptron": "MLP"}


def export(fig, name):
    for suffix in ("tiff", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"} if suffix == "tiff" else None)
    plt.close(fig)


def figure_replication(report):
    per_station = report["per_station"]
    stations = sorted(per_station, key=lambda s: per_station[s]["Multiple Linear Regression"]["MAE"])
    positions = np.arange(len(stations))

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.8), height_ratios=[1.15, 1])

    ax = axes[0]
    width = 0.26
    for offset, name in zip((-width, 0, width), COLORS):
        values = [per_station[s][name]["MAE"] for s in stations]
        ax.bar(positions + offset, values, width=width, color=COLORS[name], label=SHORT[name])
    climatology = [per_station[s]["Climatology"]["MAE"] for s in stations]
    ax.plot(positions, climatology, color="#1a1a1a", ls="--", lw=0.8, marker="_",
            markersize=6, label="Climatology")
    ax.set_xticks(positions)
    ax.set_xticklabels([s.replace("_", " ") for s in stations], rotation=45, ha="right")
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(a) Replication across 11 independent series", loc="left")
    ax.legend(frameon=False, ncol=4, columnspacing=0.7, handlelength=1.1, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(climatology) * 1.35)

    ax = axes[1]
    curve = report["pooled_learning_curve"]
    for name in COLORS:
        sizes = [point["n_train"] for point in curve[name]]
        values = [point["MAE"] for point in curve[name]]
        deviations = [point["sd"] for point in curve[name]]
        ax.plot(sizes, values, marker="o", markersize=2.4, color=COLORS[name], label=SHORT[name])
        ax.fill_between(sizes, np.array(values) - np.array(deviations),
                        np.array(values) + np.array(deviations), color=COLORS[name], alpha=0.15, lw=0)
    ax.axhline(report["pooled"]["Climatology"]["MAE"], color="#8c8c8c", ls="--", lw=0.8,
               label="Climatology")
    ax.set_xscale("log")
    ax.set_xlabel("Pooled training months (log scale)")
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(b) Learning curves up to 2,475 training months", loc="left")
    ax.legend(frameon=False, ncol=4, columnspacing=0.7, handlelength=1.2)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout(pad=0.3)
    export(fig, "figure8_multistation_replication")


def figure_nasa_validation(validation):
    monthly = pd.read_csv(RESULTS / "nasa_vs_senamhi_monthly.csv", parse_dates=["ym"])

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.75))

    ax = axes[0]
    ax.plot(monthly["ym"], monthly["tmean_senamhi"], color="#1a1a1a", lw=0.7, label="SENAMHI")
    ax.plot(monthly["ym"], monthly["tmean_nasa"], color="#cc0000", lw=0.7, label="NASA POWER")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Year")
    ax.set_title("(a) Monthly series", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    senamhi_anomaly = monthly["tmean_senamhi"] - monthly["tmean_senamhi"].mean()
    nasa_anomaly = monthly["tmean_nasa"] - monthly["tmean_nasa"].mean()
    ax.scatter(senamhi_anomaly, nasa_anomaly, s=4, color="#0072b2", alpha=0.75, edgecolors="none")
    limits = [min(senamhi_anomaly.min(), nasa_anomaly.min()) - 0.3,
              max(senamhi_anomaly.max(), nasa_anomaly.max()) + 0.3]
    ax.plot(limits, limits, color="#1a1a1a", lw=0.5, ls="--")
    ax.text(0.05, 0.94, f"r = {validation['monthly_tmean']['anomaly_correlation']:.3f}",
            transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlabel("SENAMHI anomaly (°C)")
    ax.set_ylabel("NASA anomaly (°C)")
    ax.set_title("(b) Anomalies", loc="left", fontsize=6)
    ax.grid(alpha=0.25)

    fig.tight_layout(pad=0.25, w_pad=0.6)
    export(fig, "figure9_nasa_validation")


def main():
    report = json.loads((RESULTS / "13_multistation_replication.json").read_text(encoding="utf-8"))
    validation = json.loads((RESULTS / "12_nasa_validation.json").read_text(encoding="utf-8"))

    figure_replication(report)
    figure_nasa_validation(validation["validation_against_senamhi"])

    for path in sorted(FIGURES.glob("figure[89]*.tiff")):
        print(" ", path.name, f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
