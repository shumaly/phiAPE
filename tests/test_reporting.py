from __future__ import annotations

import json

import pandas as pd

from phiape_benchmarks import reporting
from phiape_benchmarks.protocol import PROTOCOL
from phiape_benchmarks.runner import metadata_path, results_path, save_results
from test_statistics import synthetic_runs


def test_complete_report_and_workbook_schema(tmp_path, monkeypatch) -> None:
    datasets = tuple(f"Dataset{i}" for i in range(8))
    monkeypatch.setattr("phiape_benchmarks.statistics.dataset_names", lambda archive: datasets)
    output_root = tmp_path / "outputs"
    for archive in reporting.ARCHIVES:
        for dimension in reporting.DIMENSIONS:
            save_results(synthetic_runs(datasets, dimension), results_path(output_root, archive, dimension))
            metadata = {
                "archive": archive,
                "dimension": dimension,
                "methods": list(reporting.METHODS),
                "protocol": PROTOCOL.to_dict(),
                "environment": {"device": "cpu", "python": "test"},
            }
            path = metadata_path(output_root, archive, dimension)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metadata), encoding="utf-8")

    workbook = reporting.generate_report(output_root)
    assert workbook.exists()
    expected_sheets = {
        "Protocol",
        "UEA d32 Runs",
        "UEA d128 Runs",
        "UCR d32 Runs",
        "UCR d128 Runs",
        "UEA Means",
        "UCR Means",
        "Method Summary",
        "Friedman",
        "Wilcoxon Holm",
    }
    assert set(pd.ExcelFile(workbook).sheet_names) == expected_sheets
    report_directory = output_root / "reports"
    assert len(list(report_directory.glob("*.png"))) == 4
    assert len(pd.read_csv(report_directory / "phiAPE_wilcoxon_holm.csv")) == 40
    assert pd.read_csv(report_directory / "phiAPE_friedman_tests.csv")["reject_equal_performance"].all()
