"""Histogram and kernel density plots for analyzer reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.plot.generate_plots import FIGURE_DPI, FIGURE_SIZE
from src.utilities import log

from .generate_report import _metric_unit
from .statistical_analysis import METRICS, remove_outliers_zscore

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def plot_distribution(
    values: list[float], metric: str, variant: str, unit: str, filename: Path
) -> None:
    """Save a histogram with a kernel density estimate when one is available."""
    sample = np.asarray(values, dtype=float)
    if not np.isfinite(sample).all():
        log.warn(
            f"Skipping distribution plot for {metric} ({variant}): non-finite values."
        )
        return
    if not len(sample):
        log.warn(
            f"Skipping distribution plot for {metric} ({variant}): no measurements."
        )
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    try:
        can_estimate_density = len(sample) >= 2 and np.ptp(sample) > 0
        ax.hist(
            sample,
            bins="auto",
            density=can_estimate_density,
            color="lightblue",
            edgecolor="black",
            label="Measurements",
        )
        if can_estimate_density:
            try:
                kde = gaussian_kde(sample)
                x = np.linspace(sample.min(), sample.max(), 256)
                ax.plot(x, kde(x), color="red", label="Kernel density estimate")
            except np.linalg.LinAlgError:
                log.warn(
                    f"Density estimate unavailable for {metric} ({variant}): singular data."
                )
                ax.text(
                    0.02,
                    0.95,
                    "Density estimate unavailable",
                    transform=ax.transAxes,
                    va="top",
                )
        else:
            ax.text(
                0.02,
                0.95,
                "Density unavailable: fewer than two distinct values",
                transform=ax.transAxes,
                va="top",
            )
        ax.set_xlabel(f"{metric} ({unit})")
        ax.set_ylabel("Density" if can_estimate_density else "Count")
        ax.set_title(f"{metric} distribution ({variant} code smell)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)


def generate_distribution_plots(
    df_with: pd.DataFrame, df_without: pd.DataFrame, profiler: str, output_dir: Path
) -> None:
    """Plot report samples for each metric and code variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        if metric not in df_with.columns or metric not in df_without.columns:
            continue
        display_metric = (
            "co2_eq" if profiler == "carbon" and metric == "ane_mj" else metric
        )
        unit = _metric_unit(metric, display_metric)
        for variant, frame in (("with", df_with), ("without", df_without)):
            values = frame[metric].dropna().tolist()
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                log.warn(
                    f"Skipping distribution plot for {display_metric} ({variant}): "
                    "non-finite values."
                )
                continue
            values = remove_outliers_zscore(values)
            plot_distribution(
                values,
                display_metric,
                variant,
                unit,
                output_dir / f"{display_metric}_{variant}_smell_distribution.png",
            )
