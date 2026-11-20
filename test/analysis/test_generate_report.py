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
        """When samples differ significantly, the report should contain a table."""
        rng = np.random.default_rng(42)
        df_with = _make_df(rng.normal(100, 2, 50).tolist())
        df_without = _make_df(rng.normal(50, 2, 50).tolist())

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        # Table header must be present
        assert "| Metric |" in report
        assert "p-value (Mann-Whitney U)" in report
        assert "Vargha-Delaney A" in report
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

        assert (
            "| Metric | Δ median | p-value (Mann-Whitney U) | Vargha-Delaney A | Effect | Sig. |"
            not in report
        )
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

        assert "Δ median" in report
        assert "For a positive with-smell median" in report

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
# generate_pr_report - Median Values and IQR section
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestMedianValues:
    def test_median_values_section_and_table_present(self):
        df_with = _make_df([10.0] * 10, gpu=[2.0] * 10, dram=[1.0] * 10)
        df_without = _make_df([8.0] * 10, gpu=[2.0] * 10, dram=[1.0] * 10)

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "### Median Values and IQR" in report
        assert "Median (with smell)" in report
        assert "IQR (with smell)" in report
        assert "Median (without smell)" in report
        assert "IQR (without smell)" in report
        assert "`cpu_mj`" in report
        assert "10.000 mJ" in report
        assert "8.000 mJ" in report

    def test_median_values_shown_even_when_not_significant(self):
        """Medians must appear for every metric, even those without a
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
        assert "### Median Values and IQR" in report
        assert "`cpu_mj`" in report

    def test_median_values_use_time_unit_for_time_s(self):
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

    def test_median_values_use_co2_unit_for_carbon_profiler(self):
        df_with = _make_df([1.0] * 10, ane=[3.0] * 10)
        df_without = _make_df([1.0] * 10, ane=[2.0] * 10)

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "carbon", "cleaned"
        )

        assert "`co2_eq`" in report
        assert "mg CO2eq" in report

    def test_iqr_uses_co2_unit_for_carbon_profiler(self):
        df_with = pd.DataFrame({"ane_mj": [1.0, 3.0]})
        df_without = pd.DataFrame({"ane_mj": [2.0, 6.0]})
        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "carbon", "raw"
        )
        assert (
            "| `co2_eq` | 2.000 mg CO2eq | 1.000 mg CO2eq | "
            "4.000 mg CO2eq | 2.000 mg CO2eq |"
        ) in report

    def test_median_values_absent_when_no_metric_has_enough_samples(self):
        df_with = _make_df([5.0])
        df_without = _make_df([5.0])

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "### Median Values and IQR" in report
        assert "No metrics had enough samples to compute medians and IQRs." in report


@pytest.mark.unit
@pytest.mark.parametrize("writer", [MarkdownWriter, AsciidocWriter])
def test_delta_and_verdict_use_with_smell_median(writer):
    with_smell = pd.DataFrame({"cpu_mj": [53.706, 53.906] * 50})
    without_smell = pd.DataFrame({"cpu_mj": [23.509, 23.709] * 50})
    report = generate_pr_report(writer(), with_smell, without_smell, "mac", "raw")

    assert "53.806 mJ" in report
    assert "23.609 mJ" in report
    assert "+56.12%" in report
    assert "56.1% lower median energy" in report
    assert "median_without" in report
    assert "with-smell median" in report


@pytest.mark.unit
def test_zero_without_smell_median_reports_full_reduction():
    with_smell = pd.DataFrame({"cpu_mj": [9.9, 10.1] * 50})
    without_smell = pd.DataFrame({"cpu_mj": [-0.1, 0.1] * 50})
    report = generate_pr_report(
        MarkdownWriter(), with_smell, without_smell, "mac", "raw"
    )
    assert "| `cpu_mj` | +100.00% |" in report
    assert "100.0% lower median energy" in report


@pytest.mark.unit
def test_zero_with_smell_median_reports_undefined_delta():
    with_smell = pd.DataFrame({"cpu_mj": [-0.1, 0.1] * 50})
    without_smell = pd.DataFrame({"cpu_mj": [9.9, 10.1] * 50})
    report = generate_pr_report(
        MarkdownWriter(), with_smell, without_smell, "mac", "raw"
    )
    assert "| `cpu_mj` | N/A |" in report
    assert "higher median energy" in report
    assert "nan%" not in report


@pytest.mark.unit
@pytest.mark.parametrize("writer", [MarkdownWriter, AsciidocWriter])
def test_skewed_samples_use_cleaned_medians_and_iqrs(writer):
    with_smell = pd.DataFrame({"cpu_mj": [10, 11, 12, 13, 19] * 10 + [1000]})
    without_smell = pd.DataFrame({"cpu_mj": [1, 2, 3, 4, 10] * 10 + [np.nan]})
    report = generate_pr_report(writer(), with_smell, without_smell, "mac", "raw")

    assert "Median (with smell)" in report
    assert "IQR (with smell)" in report
    assert "Median (without smell)" in report
    assert "IQR (without smell)" in report
    median_row = next(
        line for line in report.splitlines() if "`cpu_mj`" in line and "12.000 mJ" in line
    )
    assert [cell.strip() for cell in median_row.split("|")[1:6]] == [
        "`cpu_mj`",
        "12.000 mJ",
        "2.000 mJ",
        "3.000 mJ",
        "2.000 mJ",
    ]
    assert "+75.00%" in report
    assert "75.0% lower median energy" in report
    assert "IQR = 75th percentile − 25th percentile" in report


@pytest.mark.unit
def test_iqr_uses_linear_percentiles_for_two_samples():
    df_with = pd.DataFrame({"time_s": [1.0, 3.0]})
    df_without = pd.DataFrame({"time_s": [2.0, 6.0]})
    report = generate_pr_report(MarkdownWriter(), df_with, df_without, "mac", "raw")
    assert "| `time_s` | 2.000 s | 1.000 s | 4.000 s | 2.000 s |" in report


@pytest.mark.unit
def test_equal_medians_do_not_claim_an_increase():
    with_smell = pd.DataFrame({"cpu_mj": [0.0] * 60 + [1.0] * 40})
    without_smell = pd.DataFrame({"cpu_mj": [0.0] * 90 + [1.0] * 10})
    report = generate_pr_report(
        MarkdownWriter(), with_smell, without_smell, "mac", "raw"
    )
    assert "no change in median energy" in report
    assert "higher median energy" not in report


@pytest.mark.unit
def test_global_execution_time_uses_cleaned_median():
    df_with = pd.DataFrame(
        {"cpu_mj": [1.0] * 5, "time_s": [0.01, 0.02, 0.03, 0.04, 0.10]}
    )
    df_without = pd.DataFrame(
        {"cpu_mj": [1.0] * 5, "time_s": [0.02, 0.03, 0.04, 0.05, 0.11]}
    )
    report = generate_pr_report(MarkdownWriter(), df_with, df_without, "mac", "raw")
    assert "| **Median Execution Time** | 30.00 ms | 40.00 ms |" in report
    assert "**Average Power**" in report


@pytest.mark.unit
def test_constant_samples_are_compared_by_rank():
    df_with = pd.DataFrame({"cpu_mj": [1.0] * 10})
    df_without = pd.DataFrame({"cpu_mj": [2.0] * 10})
    report = generate_pr_report(MarkdownWriter(), df_with, df_without, "mac", "raw")
    assert "Mann-Whitney U test was undefined" not in report
    assert "Vargha-Delaney A" in report
    assert "0.000" in report
    assert ", large)" in report
