"""Tests for the Normality (Shapiro-Wilk) section of generate_pr_report.

Kept in a separate module from test_generate_report.py, which currently fails
to collect (pre-existing import error unrelated to this change).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.generate_report import generate_pr_report
from src.analysis.utils.markdown_writer import MarkdownWriter


def _normality_section(report: str) -> str:
    start = report.index("### Normality (Shapiro-Wilk)")
    end = report.index("### Statistical Analysis")
    return report[start:end]


@pytest.mark.unit
class TestNormalitySection:
    def test_normal_samples_are_not_rejected(self):
        rng = np.random.default_rng(42)
        df_with = pd.DataFrame({"cpu_mj": rng.normal(100, 2, 50)})
        df_without = pd.DataFrame({"cpu_mj": rng.normal(50, 2, 50)})

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )
        section = _normality_section(report)

        assert "Normality rate: 100.0%" in section
        assert "Not rejected" in section
        assert "Rejected" not in section.replace("Not rejected", "")
        assert "W statistic" in section
        assert "`cpu_mj`" in section
        assert "With smell" in section
        assert "Without smell" in section

    def test_skewed_samples_are_rejected(self):
        rng = np.random.default_rng(7)
        df_with = pd.DataFrame({"cpu_mj": rng.exponential(5, 60)})
        df_without = pd.DataFrame({"cpu_mj": rng.exponential(5, 60) + 10})

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )
        section = _normality_section(report)

        assert "Rejected" in section
        assert "Normality rate: 0.0%" in section

    def test_normality_does_not_gate_welch_or_cohens_d(self):
        """Non-normal samples still produce Welch's t-test / Cohen's d results."""
        rng = np.random.default_rng(7)
        df_with = pd.DataFrame({"cpu_mj": rng.exponential(5, 60)})
        df_without = pd.DataFrame({"cpu_mj": rng.exponential(5, 60) + 10})

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )

        assert "### Statistical Analysis" in report
        assert "Cohen\u2019s d" in report

    def test_insufficient_samples_reports_no_tested_distributions(self):
        df_with = pd.DataFrame({"cpu_mj": [5.0, 5.1]})
        df_without = pd.DataFrame({"cpu_mj": [5.0, 5.1]})

        report = generate_pr_report(
            MarkdownWriter(), df_with, df_without, "mac-silicon", "cleaned"
        )
        section = _normality_section(report)

        assert (
            "No metric/group distributions had enough valid measurements "
            "to test normality." in section
        )
