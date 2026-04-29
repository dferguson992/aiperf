# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matplotlib renderer for latency-throughput uncertainty plots."""

import math

import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from aiperf.plot.constants import (
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_GREEN,
    PlotTheme,
)
from aiperf.plot.models.uncertainty import LatencyThroughputUncertaintyData

_EIGENVALUE_FLOOR = 1e-12


def _ellipse_params_from_covariance(
    point_cov_xy: float,
    x_half_width: float,
    y_half_width: float,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Compute Ellipse width, height, and angle from covariance.

    Returns:
        (width, height, angle_degrees) for matplotlib.patches.Ellipse.
    """
    var_x = x_half_width**2
    var_y = y_half_width**2
    cov = np.array([[var_x, point_cov_xy], [point_cov_xy, var_y]])

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, _EIGENVALUE_FLOOR)

    scale = math.sqrt(chi2.ppf(confidence_level, df=2))
    # width/height are full diameters (2 * semi-axis)
    width = 2.0 * scale * math.sqrt(float(eigenvalues[1]))
    height = 2.0 * scale * math.sqrt(float(eigenvalues[0]))
    angle_rad = math.atan2(float(eigenvectors[1, 1]), float(eigenvectors[0, 1]))
    angle_deg = math.degrees(angle_rad)

    return width, height, angle_deg


def _ellipse_params_axis_aligned(
    x_half_width: float,
    y_half_width: float,
) -> tuple[float, float, float]:
    """Compute Ellipse width, height for axis-aligned case.

    Returns:
        (width, height, 0.0) — full diameters, zero rotation.
    """
    return 2.0 * x_half_width, 2.0 * y_half_width, 0.0


def render_matplotlib_uncertainty(
    data: LatencyThroughputUncertaintyData,
    theme: PlotTheme = PlotTheme.LIGHT,
) -> matplotlib.figure.Figure:
    """Render uncertainty plot using Matplotlib.

    Args:
        data: Shared data contract with benchmark points and metadata.
        theme: Plot theme for NVIDIA brand styling.

    Returns:
        Matplotlib Figure object.
    """
    colors = LIGHT_THEME_COLORS if theme == PlotTheme.LIGHT else DARK_THEME_COLORS

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["paper"])

    title = data.title or "Latency vs Throughput (Joint Uncertainty)"
    x_label = data.x_label or "Latency"
    y_label = data.y_label or "Throughput"

    ax.set_title(title, color=colors["text"], fontsize=14)
    ax.set_xlabel(x_label, color=colors["text"])
    ax.set_ylabel(y_label, color=colors["text"])
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["border"])
    ax.grid(True, color=colors["grid"], alpha=0.3)

    if not data.points:
        return fig

    sorted_points = sorted(data.points, key=lambda p: p.x_mean)

    x_vals = [p.x_mean for p in sorted_points]
    y_vals = [p.y_mean for p in sorted_points]

    # Mean-point line with markers (line only when >1 point)
    linestyle = "-" if len(sorted_points) > 1 else "None"
    ax.plot(
        x_vals,
        y_vals,
        color=NVIDIA_GREEN,
        marker="o",
        markersize=6,
        linestyle=linestyle,
        linewidth=2,
        label="Mean",
        zorder=3,
    )

    # Asymmetric crosshair error bars
    xerr_lower = [p.x_mean - p.x_ci_low for p in sorted_points]
    xerr_upper = [p.x_ci_high - p.x_mean for p in sorted_points]
    yerr_lower = [p.y_mean - p.y_ci_low for p in sorted_points]
    yerr_upper = [p.y_ci_high - p.y_mean for p in sorted_points]

    ax.errorbar(
        x_vals,
        y_vals,
        xerr=[xerr_lower, xerr_upper],
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor=NVIDIA_GREEN,
        elinewidth=1,
        capsize=3,
        alpha=0.7,
        zorder=2,
    )

    # Confidence ellipses
    level_pct = int(data.confidence_level * 100)
    ellipse_added = False
    low_n_added = False

    for point in sorted_points:
        x_half = (point.x_ci_high - point.x_ci_low) / 2.0
        y_half = (point.y_ci_high - point.y_ci_low) / 2.0

        if point.cov_xy is not None and point.cov_xy != 0:
            width, height, angle = _ellipse_params_from_covariance(
                point_cov_xy=point.cov_xy,
                x_half_width=x_half,
                y_half_width=y_half,
                confidence_level=data.confidence_level,
            )
        else:
            width, height, angle = _ellipse_params_axis_aligned(
                x_half_width=x_half,
                y_half_width=y_half,
            )

        # Low-n points get dashed border and reduced opacity
        is_low_n = point.n_runs is not None and point.n_runs < 3
        linestyle = "--" if is_low_n else "-"
        alpha = 0.08 if is_low_n else 0.15

        if is_low_n and not low_n_added:
            ellipse_label = "Low sample (n < 3)"
            low_n_added = True
        elif not is_low_n and not ellipse_added:
            ellipse_label = f"{level_pct}% Confidence Region"
            ellipse_added = True
        else:
            ellipse_label = None

        ellipse = matplotlib.patches.Ellipse(
            xy=(point.x_mean, point.y_mean),
            width=width,
            height=height,
            angle=angle,
            facecolor=NVIDIA_GREEN,
            edgecolor=NVIDIA_GREEN,
            alpha=alpha,
            linewidth=1,
            linestyle=linestyle,
            label=ellipse_label,
            zorder=1,
        )
        ax.add_patch(ellipse)

    # Text annotations for labeled points
    for point in sorted_points:
        if point.label is not None:
            ax.annotate(
                point.label,
                (point.x_mean, point.y_mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color=colors["text"],
                fontsize=8,
            )

    ax.legend(
        facecolor=colors["paper"], edgecolor=colors["border"], labelcolor=colors["text"]
    )

    fig.tight_layout()
    return fig
