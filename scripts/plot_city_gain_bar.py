"""Grouped bar chart: gain in best UE score vs. no-gate baseline, per city per gate config.

Expected input CSV (default: city_gain_data.csv), long format:
    city, config, gain[, N]
where config is one of "fixed", "rolling_mean", "calibration".
An optional N column (instance size) can be used to order cities ascending;
otherwise cities are sorted alphabetically.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_ORDER = ["fixed", "rolling_mean", "calibration"]
CONFIG_LABELS = {
    "fixed": "Fixed threshold",
    "rolling_mean": "Rolling mean",
    "calibration": "Calibration",
}
CONFIG_COLORS = {
    "fixed": "#2a78d6",       # blue
    "rolling_mean": "#eb6834",  # orange
    "calibration": "#1baf7a",   # aqua
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_data(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)

    print(f"[plot_city_gain_bar] '{csv_path}' not found; using placeholder sample data "
          f"(includes known sanity-check values).", file=sys.stderr)
    cities = ["Vicenza", "Verona", "Monza", "Genova", "Palermo", "Torino",
              "Milano", "Napoli", "Roma", "Firenze", "Madonnetta", "Trieste"]
    rng = np.random.default_rng(0)
    known = {
        ("Monza", "calibration"): 2.03,
        ("Torino", "calibration"): 24.2,
        ("Palermo", "calibration"): 6.5,
        ("Firenze", "calibration"): -1.14,
        ("Madonnetta", "calibration"): -0.28,
        ("Trieste", "fixed"): 0.0,
        ("Trieste", "rolling_mean"): 0.0,
        ("Trieste", "calibration"): 0.0,
    }
    rows = []
    for city in cities:
        for config in CONFIG_ORDER:
            gain = known.get((city, config))
            if gain is None:
                gain = rng.normal(loc=3.0, scale=6.0)
            rows.append({"city": city, "config": config, "gain": gain})
    return pd.DataFrame(rows)


def order_cities(df: pd.DataFrame) -> list:
    if "N" in df.columns:
        return (
            df.groupby("city")["N"].first().sort_values().index.tolist()
        )
    return sorted(df["city"].unique())


def plot(df: pd.DataFrame, out_path: Path) -> None:
    cities = order_cities(df)
    pivot = df.pivot_table(index="city", columns="config", values="gain")
    pivot = pivot.reindex(index=cities, columns=CONFIG_ORDER)

    n_groups = len(cities)
    n_bars = len(CONFIG_ORDER)
    bar_width = 0.24
    gap = 0.02
    group_width = n_bars * bar_width + (n_bars - 1) * gap
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, config in enumerate(CONFIG_ORDER):
        offset = (i - (n_bars - 1) / 2) * (bar_width + gap)
        ax.bar(
            x + offset,
            pivot[config].values,
            width=bar_width,
            label=CONFIG_LABELS[config],
            color=CONFIG_COLORS[config],
            edgecolor="none",
            zorder=3,
        )

    ax.axhline(0, color=BASELINE, linestyle="--", linewidth=1.2, zorder=2)

    ax.set_ylabel("Gain in best UE score vs. no-gate baseline", color=INK_PRIMARY, fontsize=11)
    ax.set_title(
        "Gain in best UE score vs. no-gate baseline across twelve Italian cities",
        color=INK_PRIMARY, fontsize=13, pad=14,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha="right", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=10)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color=GRIDLINE, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    legend = ax.legend(
        title="Gate configuration", frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), fontsize=10,
    )
    legend.get_title().set_color(INK_PRIMARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[plot_city_gain_bar] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="city_gain_data.csv", type=Path,
                         help="Path to input CSV with columns city, config, gain[, N]")
    parser.add_argument("--out", default="city_gain_bar.png", type=Path,
                         help="Output PNG path")
    args = parser.parse_args()

    df = load_data(args.csv)
    plot(df, args.out)


if __name__ == "__main__":
    main()
