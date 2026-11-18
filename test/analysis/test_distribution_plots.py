"""Tests for analyzer histogram and density plots."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

from src.analysis.distribution_plots import (
    generate_distribution_plots,
    plot_distribution,
)


@pytest.mark.integration
def test_histogram_and_density_use_measurements(tmp_path):
    path = tmp_path / "distribution.png"
    values = [1.0, 2.0, 2.5, 4.0, 5.0]
    with patch(
        "src.analysis.distribution_plots.gaussian_kde", wraps=gaussian_kde
    ) as kde:
        plot_distribution(values, "cpu_mj", "with", "mJ", path)

    np.testing.assert_array_equal(kde.call_args.args[0], values)
    assert path.read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == []


@pytest.mark.integration
@pytest.mark.parametrize("values", [[3.0], [3.0, 3.0, 3.0]])
def test_constant_samples_produce_histograms_without_density(tmp_path, values):
    path = tmp_path / "constant.png"
    with patch("src.analysis.distribution_plots.gaussian_kde") as kde:
        plot_distribution(values, "cpu_mj", "without", "mJ", path)
    kde.assert_not_called()
    assert path.read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == []


@pytest.mark.unit
@pytest.mark.parametrize("values", [[], [1.0, float("inf")]])
def test_unusable_samples_are_logged_and_skipped(tmp_path, values):
    path = tmp_path / "unavailable.png"
    with patch("src.analysis.distribution_plots.log.warn") as warn:
        plot_distribution(values, "cpu_mj", "with", "mJ", path)
    warn.assert_called_once()
    assert not path.exists()


@pytest.mark.integration
def test_generated_plots_are_separate_for_each_variant(tmp_path):
    with_smell = pd.DataFrame({"cpu_mj": [1.0, 2.0, 3.0]})
    without_smell = pd.DataFrame({"cpu_mj": [4.0, 5.0, 6.0]})
    generate_distribution_plots(with_smell, without_smell, "mac", tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "cpu_mj_with_smell_distribution.png",
        "cpu_mj_without_smell_distribution.png",
    }
    for path in tmp_path.iterdir():
        assert path.read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == []
