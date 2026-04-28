# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for PlotGenerator confidence methods.

Feature: confidence-interval-plots
"""

import re

import pandas as pd
import plotly.graph_objects as go
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.plot.constants import OUTLIER_RED
from aiperf.plot.core.plot_generator import PlotGenerator

# --- Shared strategies ---

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False
)
small_positive_floats = st.floats(
    min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def ci_data_points(draw: st.DrawFn, min_size: int = 1, max_size: int = 10) -> dict:
    """Generate lists of x, y, ci_low, ci_high values."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    y_values = [draw(finite_floats) for _ in range(n)]
    spreads = [draw(positive_floats) for _ in range(n)]
    return {
        "x": [draw(finite_floats) for _ in range(n)],
        "y": y_values,
        "y_ci_low": [y - s for y, s in zip(y_values, spreads, strict=False)],
        "y_ci_high": [y + s for y, s in zip(y_values, spreads, strict=False)],
        "n": n,
    }


@st.composite
def ci_data_points_with_x_ci(draw: st.DrawFn) -> dict:
    """Generate data points with both x and y confidence intervals."""
    base = draw(ci_data_points())
    x_spreads = [draw(positive_floats) for _ in range(base["n"])]
    base["x_ci_low"] = [x - s for x, s in zip(base["x"], x_spreads, strict=False)]
    base["x_ci_high"] = [x + s for x, s in zip(base["x"], x_spreads, strict=False)]
    return base


@st.composite
def cv_data_points(draw: st.DrawFn, min_size: int = 1, max_size: int = 10) -> dict:
    """Generate lists of x, y, cv values (cv can be None)."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    cv_values = [draw(st.one_of(st.none(), small_positive_floats)) for _ in range(n)]
    return {
        "x": [draw(finite_floats) for _ in range(n)],
        "y": [draw(finite_floats) for _ in range(n)],
        "cv": cv_values,
        "n": n,
    }


@st.composite
def confidence_band_df(
    draw: st.DrawFn, n_groups: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Generate a DataFrame suitable for confidence band plotting."""
    if n_groups is None:
        n_groups = draw(st.integers(min_value=1, max_value=5))
    group_names = [f"group_{i}" for i in range(n_groups)]
    rows = []
    for g in group_names:
        n_points = draw(st.integers(min_value=2, max_value=6))
        for _ in range(n_points):
            y_val = draw(finite_floats)
            spread = draw(positive_floats)
            rows.append(
                {
                    "experiment_group": g,
                    "x_metric": draw(positive_floats),
                    "y_metric": y_val,
                    "y_metric_ci_low": y_val - spread,
                    "y_metric_ci_high": y_val + spread,
                }
            )
    df = pd.DataFrame(rows)
    return df, group_names


# --- Property 3: Error bars rendered on axes with confidence data ---


class TestErrorBarsRenderedWithConfidenceData:
    """Property 3: Error bars rendered on axes with confidence data.

    For any plot with confidence data, the generated figure should contain
    error bar configuration on the y-axis trace. When confidence data is
    available for both axes, error bars should appear on both.

    **Validates: Requirements 2.1, 2.2**
    """

    @given(data=ci_data_points())
    @settings(max_examples=100)
    def test_add_error_bars_creates_y_error_bars(self, data: dict) -> None:
        """Y-axis error bars are present when add_error_bars is called."""
        gen = PlotGenerator()
        fig = go.Figure()
        gen.add_error_bars(
            fig,
            x_values=data["x"],
            y_values=data["y"],
            y_ci_low=data["y_ci_low"],
            y_ci_high=data["y_ci_high"],
        )

        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.error_y is not None
        assert trace.error_y.visible is True
        assert trace.error_y.type == "data"

    @given(data=ci_data_points_with_x_ci())
    @settings(max_examples=100)
    def test_add_error_bars_creates_both_axis_error_bars(self, data: dict) -> None:
        """Both x and y error bars are present when both CI bounds provided."""
        gen = PlotGenerator()
        fig = go.Figure()
        gen.add_error_bars(
            fig,
            x_values=data["x"],
            y_values=data["y"],
            y_ci_low=data["y_ci_low"],
            y_ci_high=data["y_ci_high"],
            x_ci_low=data["x_ci_low"],
            x_ci_high=data["x_ci_high"],
        )

        trace = fig.data[0]
        assert trace.error_y is not None
        assert trace.error_y.visible is True
        assert trace.error_x is not None
        assert trace.error_x.visible is True


# --- Property 5: Confidence band plot contains mean line and filled region ---


class TestConfidenceBandContainsMeanLineAndFilledRegion:
    """Property 5: Confidence band plot contains mean line and filled region.

    For any valid confidence data, the confidence band handler should produce
    a figure with at least one line trace (mean) and one filled area trace (CI band).

    **Validates: Requirements 3.1**
    """

    @given(data=confidence_band_df(n_groups=1))
    @settings(max_examples=100)
    def test_single_group_has_mean_line_and_fill(
        self, data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """Single group produces at least one line trace and one filled trace."""
        df, _groups = data
        gen = PlotGenerator()
        fig = gen.create_confidence_band_plot(
            df=df,
            x_metric="x_metric",
            y_metric="y_metric",
            confidence_data={},
            group_by="experiment_group",
        )

        line_traces = [
            t
            for t in fig.data
            if getattr(t, "mode", None) == "lines"
            and t.line is not None
            and getattr(t.line, "width", 0) > 0
        ]
        fill_traces = [t for t in fig.data if getattr(t, "fill", None) == "tonexty"]

        assert len(line_traces) >= 1, "Expected at least one mean line trace"
        assert len(fill_traces) >= 1, "Expected at least one filled region trace"


# --- Property 6: Distinct confidence bands per experiment group ---


class TestDistinctConfidenceBandsPerGroup:
    """Property 6: Distinct confidence bands per experiment group.

    For N distinct groups, the figure should contain N line traces and N filled
    area traces, each with a distinct color.

    **Validates: Requirements 3.2**
    """

    @given(data=confidence_band_df())
    @settings(max_examples=100)
    def test_n_groups_produce_n_lines_and_n_fills(
        self, data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """N groups produce N mean line traces and N filled region traces."""
        df, groups = data
        n = len(groups)
        gen = PlotGenerator()
        fig = gen.create_confidence_band_plot(
            df=df,
            x_metric="x_metric",
            y_metric="y_metric",
            confidence_data={},
            group_by="experiment_group",
        )

        # Each group produces 3 traces: upper bound, lower+fill, mean line
        mean_line_traces = [
            t
            for t in fig.data
            if getattr(t, "mode", None) == "lines"
            and t.line is not None
            and getattr(t.line, "width", 0) > 0
            and t.showlegend is True
        ]
        fill_traces = [t for t in fig.data if getattr(t, "fill", None) == "tonexty"]

        assert len(mean_line_traces) == n, (
            f"Expected {n} mean lines, got {len(mean_line_traces)}"
        )
        assert len(fill_traces) == n, (
            f"Expected {n} fill traces, got {len(fill_traces)}"
        )

        # Distinct colors on mean lines
        colors = [t.line.color for t in mean_line_traces]
        assert len(set(colors)) == n, (
            f"Expected {n} distinct colors, got {len(set(colors))}"
        )


# --- Property 7: Confidence band fill opacity within range ---


class TestConfidenceBandFillOpacityWithinRange:
    """Property 7: Confidence band fill opacity within range.

    For any filled area trace, the fill opacity should be between 0.15 and 0.3.

    **Validates: Requirements 3.3**
    """

    @given(
        data=confidence_band_df(n_groups=1),
        requested_opacity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_fill_opacity_clamped_to_valid_range(
        self,
        data: tuple[pd.DataFrame, list[str]],
        requested_opacity: float,
    ) -> None:
        """Fill opacity is always between 0.15 and 0.3 regardless of input."""
        df, _groups = data
        gen = PlotGenerator()
        fig = gen.create_confidence_band_plot(
            df=df,
            x_metric="x_metric",
            y_metric="y_metric",
            confidence_data={},
            group_by="experiment_group",
            fill_opacity=requested_opacity,
        )

        fill_traces = [t for t in fig.data if getattr(t, "fill", None) == "tonexty"]

        for trace in fill_traces:
            fillcolor = trace.fillcolor
            assert fillcolor is not None
            # Extract opacity from rgba string
            match = re.search(r"rgba\(\d+,\s*\d+,\s*\d+,\s*([\d.]+)\)", fillcolor)
            assert match is not None, f"Could not parse fillcolor: {fillcolor}"
            opacity = float(match.group(1))
            assert 0.15 <= opacity <= 0.3, f"Opacity {opacity} outside [0.15, 0.3]"


# --- Property 8: CV annotation count matches data points ---


class TestCVAnnotationCountMatchesDataPoints:
    """Property 8: CV annotation count matches data points.

    For N data points with CV values (non-None), the figure should contain
    exactly N text annotations.

    **Validates: Requirements 4.1**
    """

    @given(data=cv_data_points())
    @settings(max_examples=100)
    def test_annotation_count_equals_non_none_cv_count(self, data: dict) -> None:
        """Number of annotations equals number of non-None CV values."""
        gen = PlotGenerator()
        fig = go.Figure()
        gen.add_cv_annotations(
            fig,
            x_values=data["x"],
            y_values=data["y"],
            cv_values=data["cv"],
        )

        non_none_count = sum(1 for cv in data["cv"] if cv is not None)
        annotations = fig.layout.annotations
        assert len(annotations) == non_none_count


# --- Property 9: CV annotation format ---


class TestCVAnnotationFormat:
    """Property 9: CV annotation format.

    For any CV float value, the annotation text should match "CV: {value:.1f}%".

    **Validates: Requirements 4.2**
    """

    @given(
        cv=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=100)
    def test_cv_annotation_text_format(self, cv: float) -> None:
        """CV annotation text matches 'CV: X.X%' format."""
        gen = PlotGenerator()
        fig = go.Figure()
        gen.add_cv_annotations(
            fig,
            x_values=[1.0],
            y_values=[1.0],
            cv_values=[cv],
        )

        assert len(fig.layout.annotations) == 1
        text = fig.layout.annotations[0].text
        expected = f"CV: {cv * 100:.1f}%"
        assert text == expected, f"Expected '{expected}', got '{text}'"


# --- Property 10: CV annotation color determined by threshold ---


class TestCVAnnotationColorDeterminedByThreshold:
    """Property 10: CV annotation color determined by threshold.

    If cv > threshold, annotation uses warning color.
    If cv <= threshold, annotation uses neutral color.

    **Validates: Requirements 4.3, 4.4**
    """

    @given(
        cv=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_cv_color_matches_threshold_comparison(
        self, cv: float, threshold: float
    ) -> None:
        """Warning color when cv > threshold, neutral color otherwise."""
        warning_color = "#FF0000"
        neutral_color = "#00FF00"

        gen = PlotGenerator()
        fig = go.Figure()
        gen.add_cv_annotations(
            fig,
            x_values=[1.0],
            y_values=[1.0],
            cv_values=[cv],
            threshold=threshold,
            warning_color=warning_color,
            neutral_color=neutral_color,
        )

        assert len(fig.layout.annotations) == 1
        annotation_color = fig.layout.annotations[0].font.color

        if cv > threshold:
            assert annotation_color == warning_color, (
                f"cv={cv} > threshold={threshold}, expected warning color"
            )
        else:
            assert annotation_color == neutral_color, (
                f"cv={cv} <= threshold={threshold}, expected neutral color"
            )

    @given(
        data=cv_data_points(),
        threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_cv_default_colors_applied(self, data: dict, threshold: float) -> None:
        """Default warning/neutral colors are applied when not specified."""
        gen = PlotGenerator()
        fig = go.Figure()
        gen.add_cv_annotations(
            fig,
            x_values=data["x"],
            y_values=data["y"],
            cv_values=data["cv"],
            threshold=threshold,
        )

        non_none_cvs = [cv for cv in data["cv"] if cv is not None]
        for i, cv in enumerate(non_none_cvs):
            annotation_color = fig.layout.annotations[i].font.color
            if cv > threshold:
                assert annotation_color == OUTLIER_RED
            else:
                assert annotation_color == gen.colors["secondary"]


# --- Property 8: Annotate values adds text annotations to data points ---


class TestAnnotateValuesAddsAnnotations:
    """Property 8: annotate_values adds summary annotations per group.

    When annotate_values=True, the confidence band plot should add 3 text
    annotations per group (mean, CI high, CI low) and a diamond marker trace
    for the mean point.

    **Validates: Requirements 2.1**
    """

    @given(data=confidence_band_df(n_groups=1))
    @settings(max_examples=50)
    def test_annotate_values_adds_annotations(
        self, data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """annotate_values=True produces 3 annotations per group."""
        df, groups = data
        gen = PlotGenerator()
        fig = gen.create_confidence_band_plot(
            df=df,
            x_metric="x_metric",
            y_metric="y_metric",
            confidence_data={},
            group_by="experiment_group",
            annotate_values=True,
        )

        n_groups = len(groups)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert len(annotations) == 3 * n_groups, (
            f"Expected {3 * n_groups} annotations (3 per group), got {len(annotations)}"
        )

        texts = [a.text for a in annotations]
        assert any("mean:" in t for t in texts), "Expected a mean annotation"
        assert any("CI high:" in t for t in texts), "Expected a CI high annotation"
        assert any("CI low:" in t for t in texts), "Expected a CI low annotation"

        diamond_traces = [
            t
            for t in fig.data
            if getattr(t, "mode", None) == "markers"
            and t.marker is not None
            and getattr(t.marker, "symbol", None) == "diamond"
        ]
        assert len(diamond_traces) >= 1, "Expected diamond marker for mean point"

    @given(data=confidence_band_df(n_groups=1))
    @settings(max_examples=50)
    def test_no_annotations_when_disabled(
        self, data: tuple[pd.DataFrame, list[str]]
    ) -> None:
        """annotate_values=False (default) produces no annotations."""
        df, _groups = data
        gen = PlotGenerator()
        fig = gen.create_confidence_band_plot(
            df=df,
            x_metric="x_metric",
            y_metric="y_metric",
            confidence_data={},
            group_by="experiment_group",
            annotate_values=False,
        )

        annotations = [a for a in fig.layout.annotations if a.text]
        assert len(annotations) == 0, "Expected no annotations when disabled"
