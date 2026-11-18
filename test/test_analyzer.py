from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.analyzer import (
    classify_csv_files_by_group,
    cli,
    generate_statistical_reports,
    main,
    merge_and_save_csv_groups,
    process_csv_files,
)

# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("path", ["output", "non_existent_dir"])
@pytest.mark.parametrize("verbose", [False, True])
def test_analyzer_cli_with_args(path, verbose):
    with (
        patch("src.analyzer.parse_arguments") as mock_parse,
        patch("src.analyzer.main") as mock_main,
    ):
        mock_parse.return_value = MagicMock(path=path, verbose=verbose)
        mock_main.return_value = 0
        result = cli()

    if not Path(path).exists():
        mock_main.assert_not_called()
        assert result == 1
    else:
        mock_main.assert_called_once()
        assert result == 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("path", ["output", "non_existent_dir"])
@pytest.mark.parametrize("output_format", ["markdown", "markdown"])
@pytest.mark.parametrize("verbose", [False, True])
def test_analyzer_main_with_args(path, output_format, verbose):
    with patch("src.analyzer.process_csv_files") as mock_process:
        args = MagicMock(path=path, output_format=output_format, verbose=verbose)
        result = main(args)

    mock_process.assert_called_once_with(Path(path), output_format, verbose=verbose)
    assert result == 0


@pytest.mark.unit
def test_analyzer_main_keyboard_interrupt_returns_zero():
    args = MagicMock(path="output", output_format="markdown", verbose=False)
    with patch("src.analyzer.process_csv_files", side_effect=KeyboardInterrupt):
        assert main(args) == 0


@pytest.mark.unit
def test_analyzer_main_unexpected_exception_returns_one():
    args = MagicMock(path="output", output_format="markdown", verbose=False)
    with patch("src.analyzer.process_csv_files", side_effect=RuntimeError("boom")):
        assert main(args) == 1


@pytest.mark.unit
def test_analyzer_cli_keyboard_interrupt_returns_zero():
    with patch("src.analyzer.parse_arguments", side_effect=KeyboardInterrupt):
        assert cli() == 0


@pytest.mark.unit
def test_analyzer_cli_unexpected_exception_returns_one():
    with patch("src.analyzer.parse_arguments", side_effect=RuntimeError("boom")):
        assert cli() == 1


# ---------------------------------------------------------------------------
# classify_csv_files_by_group
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_csv_files(tmp_path):
    # Create a set of mock CSV files with various paths
    csv_files = [
        tmp_path / "output/profiler1/cleaned/with_smell_run1.csv",
        tmp_path / "output/profiler1/cleaned/without_smell_run1.csv",
        tmp_path / "output/profiler2/raw/with_smell_run1.csv",
        tmp_path / "output/profiler2/raw/without_smell_run1.csv",
        tmp_path / "invalid/path/file5.csv",  # Invalid file (no profiler segment)
    ]
    for file in csv_files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    return csv_files


@pytest.mark.unit
def test_classify_csv_files_by_group_valid_files(mocked_csv_files, tmp_path):
    result = classify_csv_files_by_group(mocked_csv_files, tmp_path / "output")

    assert ("profiler1", "cleaned", "with_smell") in result
    assert ("profiler1", "cleaned", "without_smell") in result
    assert ("profiler2", "raw", "with_smell") in result
    assert ("profiler2", "raw", "without_smell") in result
    assert len(result) == 4


@pytest.mark.unit
def test_classify_csv_files_by_group_skips_invalid_path(mocked_csv_files, tmp_path):
    # The fixture contains one file under "invalid/path/" - it must be skipped
    result = classify_csv_files_by_group(mocked_csv_files, tmp_path / "output")

    all_files = [f for files in result.values() for f in files]
    assert not any("invalid" in str(f) for f in all_files)


@pytest.mark.unit
def test_classify_csv_files_by_group_empty_input(tmp_path):
    assert classify_csv_files_by_group([], tmp_path / "output") == {}


@pytest.mark.unit
def test_classify_csv_files_by_group_no_data_type_segment(tmp_path):
    # File under output/<profiler>/ but neither "cleaned" nor "raw" in path
    csv_file = tmp_path / "output" / "profiler1" / "with_smell_file.csv"
    csv_file.parent.mkdir(parents=True)
    csv_file.touch()

    result = classify_csv_files_by_group([csv_file], tmp_path / "output")

    assert result == {}


@pytest.mark.unit
def test_classify_csv_files_by_group_no_smell_segment(tmp_path):
    # File has no "with_smell" or "without_smell" in its stem
    csv_file = tmp_path / "output" / "profiler1" / "cleaned" / "unknown.csv"
    csv_file.parent.mkdir(parents=True)
    csv_file.touch()

    result = classify_csv_files_by_group([csv_file], tmp_path / "output")

    assert result == {}


# ---------------------------------------------------------------------------
# merge_and_save_csv_groups
# ---------------------------------------------------------------------------

_SAMPLE_DF = pd.DataFrame(
    {
        "cpu_mj": [1.0, 2.0],
        "gpu_mj": [3.0, 4.0],
        "ane_mj": [0.1, 0.2],
        "dram_mj": [0.5, 0.6],
    }
)


@pytest.fixture
def csv_groups_with_content(tmp_path):
    """Two groups, each with two CSV files that share the same schema."""
    groups: dict[tuple, list] = {}
    all_files: list[Path] = []
    for smell in ("with_smell", "without_smell"):
        group_dir = tmp_path / "output" / "profiler1" / "cleaned" / smell
        group_dir.mkdir(parents=True)
        files = []
        for i in range(2):
            f = group_dir / f"run_{i}.csv"
            _SAMPLE_DF.to_csv(f, index=False)
            files.append(f)
            all_files.append(f)
        groups[("profiler1", "cleaned", smell)] = files
    return all_files, groups


@pytest.mark.unit
@pytest.mark.parametrize("verbose", [False, True])
def test_merge_and_save_csv_groups_creates_merged_files(
    csv_groups_with_content, tmp_path, verbose
):
    all_files, groups = csv_groups_with_content

    with patch("src.analyzer.ANALYSIS_DIR", tmp_path / "results"):
        result = merge_and_save_csv_groups(all_files, groups, verbose=verbose)

    assert ("profiler1", "cleaned", "with_smell") in result
    assert ("profiler1", "cleaned", "without_smell") in result

    for path in result.values():
        assert path.exists()
        df = pd.read_csv(path)
        # Each group had 2 files of 2 rows each → 4 rows merged
        assert len(df) == 4


@pytest.mark.unit
def test_merge_and_save_csv_groups_empty_input():
    result = merge_and_save_csv_groups([], {})
    assert result == {}


# ---------------------------------------------------------------------------
# generate_statistical_reports
# ---------------------------------------------------------------------------


@pytest.fixture
def merged_paths(tmp_path):
    """Minimal merged CSVs for the report generator."""
    paths: dict[tuple, Path] = {}
    for smell in ("with_smell", "without_smell"):
        csv_path = tmp_path / "cleaned" / "profiler1" / f"{smell}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _SAMPLE_DF.to_csv(csv_path, index=False)
        paths[("profiler1", "cleaned", smell)] = csv_path
    return paths


@pytest.mark.unit
@pytest.mark.parametrize("verbose", [False, True])
def test_generate_statistical_reports_writes_report(merged_paths, tmp_path, verbose):
    with (
        patch("src.analyzer.ANALYSIS_DIR", tmp_path),
        patch("src.analyzer.generate_distribution_plots") as mock_plots,
        patch(
            "src.analyzer.generate_pr_report", return_value="# report"
        ) as mock_report,
    ):
        generate_statistical_reports(merged_paths, "markdown", verbose=verbose)

    mock_report.assert_called_once()
    report_file = tmp_path / "cleaned" / "profiler1" / "profiler1_report.md"
    assert report_file.exists()
    assert report_file.read_text() == "# report"
    assert mock_plots.call_args.args[3] == report_file.parent / "distributions"


@pytest.mark.integration
@pytest.mark.parametrize("data_type", ["raw", "cleaned"])
@pytest.mark.parametrize("output_format", ["markdown", "asciidoc"])
def test_report_generates_distributions_from_report_samples(tmp_path, data_type, output_format):
    report_dir = tmp_path / data_type / "carbon"
    report_dir.mkdir(parents=True)
    paths = {}
    for smell, base in (("with_smell", 1.0), ("without_smell", 5.0)):
        path = report_dir / f"{smell}.csv"
        pd.DataFrame(
            {
                "cpu_mj": [base] * 10 + [base + 1] * 10 + [1000.0],
                "ane_mj": [base] * 21,
                "time_s": [base] * 21,
            }
        ).to_csv(path, index=False)
        paths[("carbon", data_type, smell)] = path

    with (
        patch("src.analyzer.ANALYSIS_DIR", tmp_path),
        patch("src.analysis.generate_report.get_hardware_details", return_value={}),
        patch("src.analysis.distribution_plots.plot_distribution") as plot,
    ):
        generate_statistical_reports(paths, output_format)

    assert plot.call_count == 6
    for call in plot.call_args_list:
        values, metric, variant, unit, filename = call.args
        assert filename == (
            report_dir / "distributions" / f"{metric}_{variant}_smell_distribution.png"
        )
        assert unit == {"cpu_mj": "mJ", "co2_eq": "mg CO2eq", "time_s": "s"}[metric]
        assert len(values) == (20 if metric == "cpu_mj" else 21)
        assert 1000.0 not in values

    extension = "md" if output_format == "markdown" else "asciidoc"
    assert (report_dir / f"carbon_report.{extension}").exists()


@pytest.mark.integration
def test_report_writes_distribution_pngs(merged_paths, tmp_path):
    with (
        patch("src.analyzer.ANALYSIS_DIR", tmp_path),
        patch("src.analysis.generate_report.get_hardware_details", return_value={}),
    ):
        generate_statistical_reports(merged_paths, "markdown")

    output = tmp_path / "cleaned" / "profiler1" / "distributions"
    for variant in ("with", "without"):
        plot = output / f"cpu_mj_{variant}_smell_distribution.png"
        assert plot.read_bytes().startswith(b"\x89PNG")


@pytest.mark.unit
def test_generate_statistical_reports_skips_missing_variant(tmp_path):
    # Only the with_smell variant is present
    csv_path = tmp_path / "cleaned" / "profiler1" / "with_smell.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _SAMPLE_DF.to_csv(csv_path, index=False)
    incomplete_paths = {("profiler1", "cleaned", "with_smell"): csv_path}

    with (
        patch("src.analyzer.ANALYSIS_DIR", tmp_path),
        patch("src.analyzer.generate_pr_report") as mock_report,
    ):
        generate_statistical_reports(incomplete_paths, "markdown")

    mock_report.assert_not_called()


# ---------------------------------------------------------------------------
# process_csv_files (integration - sub-functions mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("verbose", [False, True])
def test_process_csv_files_calls_pipeline(tmp_path, verbose):
    # Create a dummy CSV so the early-return guard is not triggered
    dummy_csv = tmp_path / "dummy.csv"
    dummy_csv.touch()

    fake_group = {("p", "cleaned", "with_smell"): []}
    fake_merged = {("p", "cleaned", "with_smell"): tmp_path / "x.csv"}

    with (
        patch(
            "src.analyzer.classify_csv_files_by_group", return_value=fake_group
        ) as mock_classify,
        patch(
            "src.analyzer.merge_and_save_csv_groups", return_value=fake_merged
        ) as mock_merge,
        patch("src.analyzer.generate_statistical_reports") as mock_reports,
    ):
        process_csv_files(tmp_path, "markdown", verbose=verbose)

    mock_classify.assert_called_once()
    mock_merge.assert_called_once_with(
        mock_classify.call_args[0][0], fake_group, verbose=verbose
    )
    mock_reports.assert_called_once_with(fake_merged, "markdown", verbose=verbose)
