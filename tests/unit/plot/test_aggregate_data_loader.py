# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests for AggregateDataLoader.

Feature: confidence-interval-plots
"""

import tempfile
from pathlib import Path

import orjson
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.plot.core.aggregate_data_loader import (
    AGGREGATE_JSON_FILENAME,
    AGGREGATE_SUBDIR,
    AggregateDataLoader,
    ConfidenceMetricData,
)

# --- Hypothesis strategies ---

finite_floats = st.floats(
    min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False
)

metric_names = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_0123456789"),
    min_size=1,
    max_size=40,
)

metric_units = st.sampled_from(
    ["ms", "tokens", "tokens/sec", "requests/sec", "KB", "count", "ratio", "%"]
)


@st.composite
def confidence_metric_dicts(draw: st.DrawFn) -> dict:
    """Generate a valid ConfidenceMetricData as a raw dict."""
    return {
        "mean": draw(finite_floats),
        "std": draw(positive_floats),
        "min": draw(finite_floats),
        "max": draw(finite_floats),
        "cv": draw(st.one_of(st.none(), finite_floats)),
        "se": draw(positive_floats),
        "ci_low": draw(finite_floats),
        "ci_high": draw(finite_floats),
        "t_critical": draw(positive_floats),
        "unit": draw(metric_units),
    }


@st.composite
def valid_aggregate_jsons(draw: st.DrawFn) -> dict:
    """Generate a valid aggregate JSON structure."""
    num_metrics = draw(st.integers(min_value=1, max_value=10))
    names = draw(
        st.lists(metric_names, min_size=num_metrics, max_size=num_metrics, unique=True)
    )
    metrics = {name: draw(confidence_metric_dicts()) for name in names}
    num_runs = draw(st.integers(min_value=1, max_value=20))
    run_labels = [f"run_{i:04d}" for i in range(1, num_runs + 1)]

    return {
        "schema_version": "1.0",
        "aiperf_version": "0.6.0",
        "metadata": {
            "aggregation_type": "confidence",
            "num_profile_runs": num_runs,
            "num_successful_runs": num_runs,
            "failed_runs": [],
            "confidence_level": draw(
                st.floats(
                    min_value=0.5,
                    max_value=0.999,
                    allow_nan=False,
                    allow_infinity=False,
                )
            ),
            "run_labels": run_labels,
            "cooldown_seconds": 0.0,
        },
        "metrics": metrics,
    }


def _write_aggregate_json(base_dir: Path, data: dict) -> None:
    """Write aggregate JSON to the expected path under base_dir."""
    agg_dir = base_dir / AGGREGATE_SUBDIR
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / AGGREGATE_JSON_FILENAME).write_bytes(orjson.dumps(data))


# --- Property 1: Aggregate JSON parsing preserves all metric fields ---


class TestAggregateJsonParsingPreservesAllMetricFields:
    """Property 1: Aggregate JSON parsing preserves all metric fields.

    For any valid aggregate JSON, parsing via try_load() should produce an
    AggregateConfidenceData where every metric key is present and field values
    match the source.

    **Validates: Requirements 1.1, 1.4**
    """

    @given(data=valid_aggregate_jsons())
    @settings(max_examples=100)
    def test_try_load_preserves_metric_keys_and_values(self, data: dict) -> None:
        """All metric keys from source JSON are present with correct field values."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_aggregate_json(tmp_path, data)

            loader = AggregateDataLoader()
            result = loader.try_load(tmp_path)

            assert result is not None
            assert set(result.metrics.keys()) == set(data["metrics"].keys())

            for name, source in data["metrics"].items():
                parsed = result.metrics[name]
                assert isinstance(parsed, ConfidenceMetricData)
                assert parsed.mean == pytest.approx(source["mean"])
                assert parsed.std == pytest.approx(source["std"])
                assert parsed.min == pytest.approx(source["min"])
                assert parsed.max == pytest.approx(source["max"])
                if source["cv"] is not None:
                    assert parsed.cv == pytest.approx(source["cv"])
                else:
                    assert parsed.cv is None
                assert parsed.se == pytest.approx(source["se"])
                assert parsed.ci_low == pytest.approx(source["ci_low"])
                assert parsed.ci_high == pytest.approx(source["ci_high"])
                assert parsed.t_critical == pytest.approx(source["t_critical"])
                assert parsed.unit == source["unit"]

    @given(data=valid_aggregate_jsons())
    @settings(max_examples=100)
    def test_try_load_preserves_metadata(self, data: dict) -> None:
        """Metadata fields (confidence_level, num_runs, run_labels) are preserved."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_aggregate_json(tmp_path, data)

            loader = AggregateDataLoader()
            result = loader.try_load(tmp_path)

            assert result is not None
            assert result.confidence_level == pytest.approx(
                data["metadata"]["confidence_level"]
            )
            assert result.num_runs == data["metadata"]["num_successful_runs"]
            assert result.run_labels == data["metadata"]["run_labels"]


# --- Property 2: Malformed aggregate JSON returns None without raising ---


malformed_json_bytes = st.one_of(
    st.just(b""),
    st.just(b"{invalid json"),
    st.just(b"null"),
    st.just(b"[]"),
    st.just(b'"just a string"'),
    st.just(b"42"),
    st.binary(min_size=1, max_size=100),
)

malformed_structures = st.one_of(
    st.just({"metrics": {"m": {"mean": 1.0}}}),
    st.just(
        {
            "metadata": {
                "confidence_level": 0.95,
                "num_successful_runs": 3,
                "run_labels": [],
            }
        }
    ),
    st.just({"metadata": "bad", "metrics": {}}),
    st.just(
        {
            "metadata": {
                "confidence_level": 0.95,
                "num_successful_runs": 3,
                "run_labels": [],
            },
            "metrics": "bad",
        }
    ),
    st.just(
        {
            "metadata": {
                "confidence_level": 0.95,
                "num_successful_runs": 3,
                "run_labels": [],
            },
            "metrics": {
                "m": {
                    "mean": "not_a_float",
                    "std": 0,
                    "min": 0,
                    "max": 0,
                    "cv": 0,
                    "se": 0,
                    "ci_low": 0,
                    "ci_high": 0,
                    "t_critical": 0,
                    "unit": "ms",
                }
            },
        }
    ),
    st.just(
        {
            "metadata": {
                "confidence_level": 0.95,
                "num_successful_runs": 3,
                "run_labels": [],
            },
            "metrics": {"m": {"mean": 1.0}},
        }
    ),
    st.just(
        {
            "metadata": {
                "confidence_level": 0.95,
                "num_successful_runs": 3,
                "run_labels": [],
            },
            "metrics": {"m": 42},
        }
    ),
    st.just({}),
)


class TestMalformedAggregateJsonReturnsNone:
    """Property 2: Malformed aggregate JSON returns None without raising.

    For any malformed JSON input, try_load() should return None and not raise.

    **Validates: Requirements 1.3**
    """

    @given(raw_bytes=malformed_json_bytes)
    @settings(max_examples=100)
    def test_malformed_bytes_returns_none(self, raw_bytes: bytes) -> None:
        """Invalid JSON bytes never raise, always return None."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            agg_dir = tmp_path / AGGREGATE_SUBDIR
            agg_dir.mkdir()
            (agg_dir / AGGREGATE_JSON_FILENAME).write_bytes(raw_bytes)

            loader = AggregateDataLoader()
            result = loader.try_load(tmp_path)
            assert result is None

    @given(structure=malformed_structures)
    @settings(max_examples=100)
    def test_malformed_structure_returns_none_or_partial(self, structure: dict) -> None:
        """Malformed JSON structures never raise."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_aggregate_json(tmp_path, structure)

            loader = AggregateDataLoader()
            result = loader.try_load(tmp_path)
            if result is not None:
                for metric in result.metrics.values():
                    assert isinstance(metric, ConfidenceMetricData)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Missing aggregate file returns None."""
        loader = AggregateDataLoader()
        result = loader.try_load(tmp_path)
        assert result is None

    def test_empty_metrics_returns_valid_data(self, tmp_path: Path) -> None:
        """Empty metrics dict produces valid data with no metrics."""
        data = {
            "metadata": {
                "confidence_level": 0.95,
                "num_successful_runs": 3,
                "run_labels": ["r1", "r2", "r3"],
            },
            "metrics": {},
        }
        _write_aggregate_json(tmp_path, data)

        loader = AggregateDataLoader()
        result = loader.try_load(tmp_path)
        assert result is not None
        assert len(result.metrics) == 0
