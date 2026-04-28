# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for PlotSpec confidence fields round-trip through config parsing.

Feature: confidence-interval-plots
"""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.plot.config import PlotConfig
from aiperf.plot.core.plot_specs import PlotSpec

# --- Hypothesis strategies ---

# cv_threshold must be non-negative per the Pydantic validator
non_negative_floats = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def confidence_preset_values(draw: st.DrawFn) -> dict:
    """Generate valid confidence field values for a YAML preset."""
    return {
        "show_confidence": draw(st.booleans()),
        "show_cv": draw(st.booleans()),
        "cv_threshold": draw(non_negative_floats),
        "annotate_values": draw(st.booleans()),
    }


def _write_config_with_confidence(
    tmp_path: Path,
    show_confidence: bool | None = None,
    show_cv: bool | None = None,
    cv_threshold: float | None = None,
    annotate_values: bool | None = None,
) -> Path:
    """Write a minimal YAML config with optional confidence fields."""
    config_file = tmp_path / "config.yaml"
    lines = [
        "visualization:",
        "  multi_run_defaults:",
        "    - test_plot",
        "  multi_run_plots:",
        "    test_plot:",
        "      type: scatter_line",
        "      x: request_latency_avg",
        "      y: request_throughput_avg",
    ]
    if show_confidence is not None:
        lines.append(f"      show_confidence: {str(show_confidence).lower()}")
    if show_cv is not None:
        lines.append(f"      show_cv: {str(show_cv).lower()}")
    if cv_threshold is not None:
        lines.append(f"      cv_threshold: {cv_threshold}")
    if annotate_values is not None:
        lines.append(f"      annotate_values: {str(annotate_values).lower()}")
    lines += [
        "  single_run_defaults: []",
        "  single_run_plots: {}",
    ]
    config_file.write_text("\n".join(lines) + "\n")
    return config_file


# --- Property 11: PlotSpec confidence fields round-trip through config parsing ---


class TestPlotSpecConfidenceFieldsRoundTrip:
    """Property 11: PlotSpec confidence fields round-trip through config parsing.

    For any valid YAML plot preset containing show_confidence, show_cv, and
    cv_threshold values, parsing via PlotConfig._preset_to_plot_spec() should
    produce a PlotSpec with those exact field values preserved. When omitted,
    defaults should be False, False, and 0.10 respectively.

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @given(values=confidence_preset_values())
    @settings(max_examples=100)
    def test_confidence_fields_round_trip_preserves_values(
        self, values: dict, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Confidence field values survive YAML -> PlotConfig -> PlotSpec round-trip."""
        tmp_path = tmp_path_factory.mktemp("conf")
        config_file = _write_config_with_confidence(
            tmp_path,
            show_confidence=values["show_confidence"],
            show_cv=values["show_cv"],
            cv_threshold=values["cv_threshold"],
            annotate_values=values["annotate_values"],
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        spec = specs[0]
        assert isinstance(spec, PlotSpec)
        assert spec.show_confidence is values["show_confidence"]
        assert spec.show_cv is values["show_cv"]
        assert spec.cv_threshold == pytest.approx(values["cv_threshold"])
        assert spec.annotate_values is values["annotate_values"]

    def test_omitted_confidence_fields_use_defaults(self, tmp_path: Path) -> None:
        """When confidence fields are omitted, defaults are False/False/0.10."""
        config_file = _write_config_with_confidence(tmp_path)

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        spec = specs[0]
        assert spec.show_confidence is False
        assert spec.show_cv is False
        assert spec.cv_threshold == pytest.approx(0.10)
        assert spec.annotate_values is False

    def test_negative_cv_threshold_rejected(self, tmp_path: Path) -> None:
        """Negative cv_threshold is rejected by Pydantic validation."""
        config_file = _write_config_with_confidence(
            tmp_path,
            show_confidence=True,
            show_cv=True,
            cv_threshold=-0.05,
        )

        config = PlotConfig(config_file)
        with pytest.raises(ValueError, match="Config validation failed"):
            config.get_multi_run_plot_specs()
