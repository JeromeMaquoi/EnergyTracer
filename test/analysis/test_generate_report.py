"""Tests for the generate_report module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.utils.markdown_writer import MarkdownWriter
from src.analysis.generate_report import generate_pr_report
from src.analysis.utils.asciidoc_writer import AsciidocWriter

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_df(cpu, gpu=None, ane=None, dram=None):
    """Build a DataFrame with the standard metric columns."""
    n = len(cpu)
    return pd.DataFrame(
        {
            "cpu_mj": cpu,
            "gpu_mj": gpu if gpu is not None else [0.0] * n,
            "ane_mj": ane if ane is not None else [0.0] * n,
            "dram_mj": dram if dram is not None else [0.0] * n,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# generate_pr_report - structure & content
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGeneratePrReport:
    def test_report_contains_title_and_context(self):
        rng = np.random.default_rng(0)
        df_with = _make_df(rng.normal(10, 1, 30).tolist())
        df_without = _make_df(rng.normal(10, 1, 30).tolist())

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "## Energy Report" in report
        assert "`mac-silicon`" in report
        assert "cleaned" in report
        assert "30 samples (with smell)" in report
        assert "30 samples (without smell)" in report
        assert "### Verdict" in report

    def test_significant_difference_produces_table(self):
        """When means differ significantly, the report should contain a table."""
        rng = np.random.default_rng(42)
        df_with = _make_df(rng.normal(100, 2, 50).tolist())
        df_without = _make_df(rng.normal(50, 2, 50).tolist())

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        # Table header must be present
        assert "| Metric |" in report
        assert "`cpu_mj`" in report
        assert "✅" in report
        # Must have a verdict about measurable differences
        assert "measurable energy differences" in report

    def test_no_significant_difference_no_statistical_table(self):
        """When there's no significant difference, no statistical table should appear."""
        rng = np.random.default_rng(7)
        # Same distribution → no significant difference
        values = rng.normal(50, 5, 30).tolist()
        df_with = _make_df(values)
        df_without = _make_df(values)

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "| Metric | Δ mean | p-value | Cohen’s d | Effect | Sig. |" not in report
        assert "No statistically significant differences" in report
        assert "does not measurably impact" in report

    def test_carbon_profiler_renames_ane_to_co2_eq(self):
        """For the 'carbon' profiler, ane_mj should be displayed as co2_eq."""
        rng = np.random.default_rng(1)
        df_with = _make_df(
            rng.normal(100, 2, 50).tolist(), ane=rng.normal(5, 0.5, 50).tolist()
        )
        df_without = _make_df(
            rng.normal(50, 2, 50).tolist(), ane=rng.normal(2, 0.5, 50).tolist()
        )

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "carbon", "cleaned"
        )

        assert "`co2_eq`" in report
        # ane_mj should NOT appear as a metric name
        assert "`ane_mj`" not in report

    def test_delta_formula_note_present_when_significant(self):
        rng = np.random.default_rng(42)
        df_with = _make_df(rng.normal(100, 2, 50).tolist())
        df_without = _make_df(rng.normal(50, 2, 50).tolist())

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "Δ mean" in report
        assert "Positive" in report

    def test_report_is_string(self):
        df = _make_df([1.0, 2.0, 3.0])
        report = generate_pr_report(MarkdownWriter(), df, df, "mac-silicon", "raw")
        assert isinstance(report, str)

    def test_missing_metric_columns_handled(self):
        """If DataFrames are missing some metric columns, no crash."""
        df_with = pd.DataFrame({"cpu_mj": [1.0, 2.0, 3.0]})
        df_without = pd.DataFrame({"cpu_mj": [1.0, 2.0, 3.0]})

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "## Energy Report" in report

    def test_too_few_samples_skips_metric(self):
        """Metrics with fewer than 2 values after cleaning are skipped."""
        df_with = _make_df([5.0])
        df_without = _make_df([5.0])

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        # Should still generate a valid report with no table
        assert "## Energy Report" in report
        assert "### Verdict" in report


# ──────────────────────────────────────────────────────────────────────────────
# generate_pr_report - Mean Values section
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestMeanValues:
    def test_mean_values_section_and_table_present(self):
        df_with = _make_df([10.0] * 10, gpu=[2.0] * 10, dram=[1.0] * 10)
        df_without = _make_df([8.0] * 10, gpu=[2.0] * 10, dram=[1.0] * 10)

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "### Mean Values" in report
        assert "Mean (with smell)" in report
        assert "Mean (without smell)" in report
        assert "`cpu_mj`" in report
        assert "10.000 mJ" in report
        assert "8.000 mJ" in report

    def test_mean_values_shown_even_when_not_significant(self):
        """Means must appear for every metric, even those without a
        significant difference (which are excluded from the Statistical
        Analysis table)."""
        rng = np.random.default_rng(7)
        values = rng.normal(50, 5, 30).tolist()
        df_with = _make_df(values)
        df_without = _make_df(values)

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "No statistically significant differences" in report
        assert "### Mean Values" in report
        assert "`cpu_mj`" in report

    def test_mean_values_use_time_unit_for_time_s(self):
        df_with = _make_df([10.0] * 10)
        df_with["time_s"] = [0.01] * 10
        df_without = _make_df([8.0] * 10)
        df_without["time_s"] = [0.012] * 10

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "`time_s`" in report
        assert "0.010 s" in report
        assert "0.012 s" in report

    def test_mean_values_use_co2_unit_for_carbon_profiler(self):
        df_with = _make_df([1.0] * 10, ane=[3.0] * 10)
        df_without = _make_df([1.0] * 10, ane=[2.0] * 10)

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "carbon", "cleaned"
        )

        assert "`co2_eq`" in report
        assert "mg CO2eq" in report

    def test_mean_values_absent_when_no_metric_has_enough_samples(self):
        df_with = _make_df([5.0])
        df_without = _make_df([5.0])

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "### Mean Values" in report
        assert "No metrics had enough samples to compute mean values." in report


@pytest.mark.unit
@pytest.mark.parametrize("writer", [MarkdownWriter, AsciidocWriter])
def test_delta_uses_without_smell_baseline_and_verdict_uses_with(writer):
    with_smell = pd.DataFrame({"cpu_mj": [53.706, 53.906] * 50})
    without_smell = pd.DataFrame({"cpu_mj": [23.509, 23.709] * 50})
    report = generate_pr_report(writer(), with_smell, without_smell, "mac", "raw")

    assert "53.806 mJ" in report
    assert "23.609 mJ" in report
    assert "+127.90%" in report
    assert "56.1% lower energy" in report
    assert "mean_without" in report
    assert "log-space" in report
    assert "0.00e+00" not in report


@pytest.mark.unit
def test_zero_without_smell_mean_reports_undefined_delta():
    with_smell = pd.DataFrame({"cpu_mj": [9.9, 10.1] * 50})
    without_smell = pd.DataFrame({"cpu_mj": [-0.1, 0.1] * 50})
    report = generate_pr_report(
        MarkdownWriter(), with_smell, without_smell, "mac", "raw"
    )
    assert "| `cpu_mj` | N/A |" in report
    assert "100.0% lower energy" in report


@pytest.mark.unit
def test_zero_with_smell_mean_keeps_delta_but_verdict_has_no_percentage():
    with_smell = pd.DataFrame({"cpu_mj": [-0.1, 0.1] * 50})
    without_smell = pd.DataFrame({"cpu_mj": [9.9, 10.1] * 50})
    report = generate_pr_report(
        MarkdownWriter(), with_smell, without_smell, "mac", "raw"
    )
    assert "| `cpu_mj` | -100.00% |" in report
    assert "higher energy" in report
    assert "nan%" not in report


@pytest.mark.unit
def test_constant_samples_report_undefined_test():
    df_with = pd.DataFrame({"cpu_mj": [1.0] * 10})
    df_without = pd.DataFrame({"cpu_mj": [2.0] * 10})
    report = generate_pr_report(MarkdownWriter(), df_with, df_without, "mac", "raw")
    assert "Welch's test was undefined for: `cpu_mj`" in report
    assert "No statistical verdict is available" in report
    assert "0.00e+00" not in report
