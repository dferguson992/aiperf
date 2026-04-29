# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run plot type handlers.

Handlers for creating comparison plots from multiple profiling runs.
"""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.constants import DEFAULT_PERCENTILE
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import PlotSpec


class BaseMultiRunHandler:
    """
    Base class for multi-run plot handlers.

    Provides common functionality for working with multi-run DataFrames.
    """

    def __init__(self, plot_generator: PlotGenerator) -> None:
        """
        Initialize the handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
        """
        self.plot_generator = plot_generator

    def _get_metric_label(
        self, metric_name: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted metric label.

        Args:
            metric_name: Name of the metric
            stat: Statistic (e.g., "avg", "p50")
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted metric label
        """
        display_name = None
        unit = ""

        if "display_names" in available_metrics or "units" in available_metrics:
            display_name = available_metrics.get("display_names", {}).get(metric_name)
            unit = available_metrics.get("units", {}).get(metric_name, "")

        if not display_name and metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")

        if display_name:
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name
        return metric_name

    def _extract_experiment_types(
        self, data: pd.DataFrame, group_by: str | None
    ) -> dict[str, str] | None:
        """
        Extract experiment types from DataFrame for experiment groups color assignment.

        Args:
            data: DataFrame with aggregated metrics
            group_by: Column name to group by

        Returns:
            Dictionary mapping group values to experiment_type, or None
        """
        if not group_by or group_by not in data.columns:
            return None

        if "experiment_type" not in data.columns:
            return None

        experiment_types = {}
        for group_val in data[group_by].unique():
            group_df = data[data[group_by] == group_val]
            experiment_types[group_val] = group_df["experiment_type"].iloc[0]

        return experiment_types

    def _extract_group_display_names(
        self, data: pd.DataFrame, group_by: str | None
    ) -> dict[str, str] | None:
        """
        Extract group display names from DataFrame for legend labels.

        Args:
            data: DataFrame with aggregated metrics
            group_by: Column name to group by

        Returns:
            Dictionary mapping group values to display names, or None
        """
        if not group_by or group_by not in data.columns:
            return None

        # Only use group_display_name when grouping by experiment_group
        # For other groupings (e.g., 'model'), use the actual group value
        if group_by != "experiment_group":
            return None

        if "group_display_name" not in data.columns:
            return None

        display_names = {}
        for group_val in data[group_by].unique():
            group_df = data[data[group_by] == group_val]
            display_names[group_val] = group_df["group_display_name"].iloc[0]

        return display_names


class ParetoHandler(BaseMultiRunHandler):
    """Handler for Pareto curve plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if Pareto plot can be generated."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Create a Pareto curve plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        experiment_types = self._extract_experiment_types(data, spec.group_by)
        group_display_names = self._extract_group_display_names(data, spec.group_by)

        return self.plot_generator.create_pareto_plot(
            df=data,
            x_metric=x_metric.name,
            y_metric=y_metric.name,
            label_by=spec.label_by,
            group_by=spec.group_by,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
            group_display_names=group_display_names,
        )


class ScatterLineHandler(BaseMultiRunHandler):
    """Handler for scatter line plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if scatter line plot can be generated."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter line plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        experiment_types = self._extract_experiment_types(data, spec.group_by)
        group_display_names = self._extract_group_display_names(data, spec.group_by)

        return self.plot_generator.create_scatter_line_plot(
            df=data,
            x_metric=x_metric.name,
            y_metric=y_metric.name,
            label_by=spec.label_by,
            group_by=spec.group_by,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
            group_display_names=group_display_names,
        )


class LatencyThroughputUncertaintyHandler(BaseMultiRunHandler):
    """Handler for latency-throughput uncertainty plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if required metric columns exist in the DataFrame."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Build LatencyThroughputUncertaintyData from DataFrame and delegate to PlotGenerator."""
        from aiperf.plot.models.uncertainty import (
            BenchmarkPoint,
            LatencyThroughputUncertaintyData,
        )

        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        x_label = self._get_metric_label(
            x_metric.name, x_metric.stat or "avg", available_metrics
        )
        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        ci_level = getattr(spec, "ci_level", 0.95)
        if ci_level not in {0.90, 0.95, 0.99}:
            ci_level = 0.95

        group_by = "concurrency" if "concurrency" in data.columns else spec.group_by
        label_by = spec.label_by

        groups: list[str | None]
        if group_by and group_by in data.columns:
            groups = sorted(data[group_by].dropna().unique())
        else:
            groups = [None]

        points: list[BenchmarkPoint] = []
        for group in groups:
            group_df = data[data[group_by] == group] if group is not None else data

            if group_df.empty:
                continue

            import numpy as np
            from scipy import stats as scipy_stats

            x_vals = group_df[x_metric.name].values.astype(float)
            y_vals = group_df[y_metric.name].values.astype(float)
            n = len(x_vals)

            x_mean = float(np.mean(x_vals))
            y_mean = float(np.mean(y_vals))

            if n >= 2:
                t_crit = scipy_stats.t.ppf((1 + ci_level) / 2, df=n - 1)
                x_se = float(np.std(x_vals, ddof=1) / np.sqrt(n))
                y_se = float(np.std(y_vals, ddof=1) / np.sqrt(n))
                x_ci_half = t_crit * x_se
                y_ci_half = t_crit * y_se
                cov_xy = None
            else:
                x_ci_half = 0.0
                y_ci_half = 0.0
                cov_xy = None

            label_val = None
            if label_by and label_by in group_df.columns:
                label_mode = group_df[label_by].dropna().mode()
                label_val = str(label_mode.iloc[0]) if not label_mode.empty else None

            points.append(
                BenchmarkPoint(
                    x_mean=x_mean,
                    y_mean=y_mean,
                    x_ci_low=x_mean - x_ci_half,
                    x_ci_high=x_mean + x_ci_half,
                    y_ci_low=y_mean - y_ci_half,
                    y_ci_high=y_mean + y_ci_half,
                    cov_xy=cov_xy,
                    label=label_val,
                    n_runs=n,
                )
            )

        uncertainty_data = LatencyThroughputUncertaintyData(
            points=points,
            confidence_level=ci_level,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
            group_by=group_by,
        )

        experiment_types = self._extract_experiment_types(data, spec.group_by)
        group_display_names = self._extract_group_display_names(data, spec.group_by)

        return self.plot_generator.create_uncertainty_plot(
            uncertainty_data,
            experiment_types=experiment_types,
            group_display_names=group_display_names,
        )


# =============================================================================
# Discovery findings for LatencyThroughputUncertaintyHandler (Task 1.1)
# =============================================================================
#
# COLUMN AVAILABILITY AFTER DataLoader.load_run / _runs_to_dataframe
# ------------------------------------------------------------------
#
# 1. Per-run aggregated stats (from profile_export_aiperf.json):
#    - Stored in RunData.aggregated["metrics"] as MetricResult objects.
#    - MetricResult (inherits JsonMetricResult) has stat fields:
#        avg, min, max, std, p1, p5, p10, p25, p50, p75, p90, p95, p99
#        (plus: unit, tag, header, count, current, sum)
#    - NO ci_low / ci_high fields on MetricResult or JsonMetricResult.
#    - NO covariance (cov_xy) field anywhere on MetricResult.
#
# 2. Multi-run DataFrame (built by MultiRunPNGExporter._runs_to_dataframe):
#    - Flattens each RunData into one row per run.
#    - Metric columns use a SINGLE stat value per metric (DEFAULT_PERCENTILE="p50",
#      falling back to "avg"). Column name = metric tag (e.g., "request_latency").
#    - Metadata columns: run_name, model, concurrency, request_count,
#      duration_seconds, experiment_type, experiment_group, endpoint_type,
#      group_display_name, plus all flattened input_config fields.
#    - NO _ci_low / _ci_high suffix columns in this DataFrame.
#    - NO covariance columns.
#
# 3. Dashboard DataFrame (built by dashboard/utils.py runs_to_dataframe):
#    - Even simpler: only x_metric, y_metric, model, concurrency, run_idx,
#      run_name, experiment_type, experiment_group columns.
#    - Uses extract_metric_value(run, metric, stat) to pull a single stat.
#    - NO CI bounds or covariance columns.
#
# 4. Aggregate confidence data (from aggregate/profile_export_aiperf_aggregate.json):
#    - Produced by ConfidenceAggregation in orchestrator/aggregation/confidence.py.
#    - Loaded by AggregateDataLoader (plot/core/aggregate_data_loader.py).
#    - ConfidenceMetricData model HAS: mean, std, min, max, cv, se,
#      ci_low, ci_high, t_critical, unit.
#    - Keys are FLATTENED: "{metric_name}_{stat_key}" e.g., "request_latency_avg",
#      "output_token_throughput_p50".
#    - HOWEVER: AggregateDataLoader is NOT currently imported or used anywhere
#      in the multi-run pipeline (not in MultiRunPNGExporter, not in dashboard).
#    - This is the ONLY source of CI bounds in the system.
#
# 5. Covariance (cov_xy):
#    - NOT computed anywhere in the codebase. No np.cov, no cross-metric
#      covariance calculation exists.
#    - Must be DERIVED at handler construction time from per-run raw values
#      if we want rotated ellipses. Otherwise, fall back to axis-aligned
#      ellipses using ci_low/ci_high from the aggregate data.
#    - To compute cov_xy, we would need per-run (metric_x, metric_y) pairs
#      and use np.cov(). The per-run values are available from RunData.aggregated
#      but are not currently collected into a cross-metric structure.
#
# IMPLICATIONS FOR LatencyThroughputUncertaintyHandler
# ----------------------------------------------------
#
# A. The handler CANNOT rely on the existing _runs_to_dataframe DataFrame
#    for CI bounds -- those columns don't exist there.
#
# B. The handler MUST either:
#    (a) Load aggregate confidence data via AggregateDataLoader.try_load()
#        to get ci_low/ci_high per flattened metric, OR
#    (b) Compute CI bounds itself from the list of RunData objects by
#        collecting per-run stat values and running the same t-distribution
#        CI calculation that ConfidenceAggregation uses.
#
# C. For covariance (cov_xy), the handler MUST compute it from per-run
#    (x_stat, y_stat) value pairs using np.cov(). This is not available
#    from any existing data source.
#
# D. The can_handle check should verify that the x and y metric columns
#    exist in the DataFrame (same pattern as ParetoHandler/ScatterLineHandler),
#    and ideally that there are >= 2 runs (needed for CI computation).
#
# E. Stat columns available on MetricResult (STAT_KEYS from common/constants.py):
#    avg, min, max, sum, p1, p5, p10, p25, p50, p75, p90, p95, p99, std
#
# F. RecordExportResultsProcessor: writes per-request JSONL records with
#    individual metric values (not aggregated stats). Not relevant for
#    multi-run CI computation.
#
# G. MetricResultsProcessor: computes per-run summary stats (MetricResult
#    with avg/p50/p99/etc.) via MetricArray.to_result(). These are the
#    values that get written to profile_export_aiperf.json and are the
#    basis for cross-run confidence intervals.
