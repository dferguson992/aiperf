# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate confidence data loader for multi-run visualization."""

from pathlib import Path

import orjson
from pydantic import Field, ValidationError

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import AIPerfBaseModel

AGGREGATE_JSON_FILENAME = "profile_export_aiperf_aggregate.json"
AGGREGATE_SUBDIR = "aggregate"


class ConfidenceMetricData(AIPerfBaseModel):
    """Confidence statistics for a single metric."""

    mean: float = Field(description="Sample mean across runs")
    std: float = Field(description="Sample standard deviation")
    min: float = Field(description="Minimum value across runs")
    max: float = Field(description="Maximum value across runs")
    cv: float | None = Field(description="Coefficient of variation (std/mean)")
    se: float = Field(description="Standard error")
    ci_low: float = Field(description="Lower bound of confidence interval")
    ci_high: float = Field(description="Upper bound of confidence interval")
    t_critical: float = Field(description="t-distribution critical value")
    unit: str = Field(description="Unit of measurement")


class AggregateConfidenceData(AIPerfBaseModel):
    """Confidence statistics loaded from aggregate JSON."""

    confidence_level: float = Field(description="Confidence level (e.g. 0.95)")
    num_runs: int = Field(description="Number of profile runs aggregated")
    run_labels: list[str] = Field(description="Labels of aggregated runs")
    metrics: dict[str, ConfidenceMetricData] = Field(
        description="Per-metric confidence statistics keyed by flattened metric name"
    )


class AggregateDataLoader(AIPerfLoggerMixin):
    """Loads aggregate confidence data from artifact directories."""

    def try_load(self, artifact_dir: Path) -> AggregateConfidenceData | None:
        """Attempt to load aggregate confidence data.

        Searches for aggregate/profile_export_aiperf_aggregate.json.
        Returns None if not found or malformed (logs warning, never raises).

        Args:
            artifact_dir: Path to the artifact directory.

        Returns:
            Parsed aggregate confidence data, or None on any error.
        """
        json_path = artifact_dir / AGGREGATE_SUBDIR / AGGREGATE_JSON_FILENAME
        if not json_path.exists():
            self.info(lambda: f"No aggregate data found at {json_path}")
            return None

        try:
            with open(json_path, "rb") as f:
                raw = orjson.loads(f.read())
        except (orjson.JSONDecodeError, OSError) as e:
            self.warning(lambda e=e: f"Failed to read aggregate JSON: {e}")
            return None

        return self._parse_aggregate(raw)

    def _parse_aggregate(self, raw: object) -> AggregateConfidenceData | None:
        """Parse raw JSON dict into AggregateConfidenceData.

        Args:
            raw: Parsed JSON dictionary.

        Returns:
            Parsed model or None if structure is invalid.
        """
        if not isinstance(raw, dict):
            self.warning("Aggregate JSON root is not a dict")
            return None

        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            self.warning("Aggregate JSON missing or invalid 'metadata' key")
            return None

        raw_metrics = raw.get("metrics")
        if not isinstance(raw_metrics, dict):
            self.warning("Aggregate JSON missing or invalid 'metrics' key")
            return None

        parsed_metrics: dict[str, ConfidenceMetricData] = {}
        for name, entry in raw_metrics.items():
            if not isinstance(entry, dict):
                self.warning(
                    lambda name=name: f"Skipping non-dict metric entry: {name}"
                )
                continue
            try:
                parsed_metrics[name] = ConfidenceMetricData(**entry)
            except ValidationError as e:
                self.warning(
                    lambda name=name, e=e: f"Skipping malformed metric '{name}': {e}"
                )

        try:
            return AggregateConfidenceData(
                confidence_level=metadata.get("confidence_level", 0.95),
                num_runs=metadata.get(
                    "num_successful_runs", metadata.get("num_profile_runs", 0)
                ),
                run_labels=metadata.get("run_labels", []),
                metrics=parsed_metrics,
            )
        except ValidationError as e:
            self.warning(
                lambda e=e: f"Failed to construct AggregateConfidenceData: {e}"
            )
            return None

    def get_metric(
        self, data: AggregateConfidenceData, name: str
    ) -> ConfidenceMetricData | None:
        """Get confidence data for a specific metric by flattened name.

        Args:
            data: The loaded aggregate confidence data.
            name: Flattened metric name (e.g. 'request_latency_avg').

        Returns:
            Confidence metric data, or None if not found.
        """
        return data.metrics.get(name)
