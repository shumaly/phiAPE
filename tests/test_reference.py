from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from phiape_benchmarks.protocol import (
    ARCHIVES,
    DIMENSIONS,
    METHODS,
    PROTOCOL,
    UCR_DATASETS,
    UEA_DATASETS,
)
from phiape_benchmarks.reference import sha256_file, validate_reference_manifest


def test_locked_dataset_lists_match_publication_rows() -> None:
    root = Path(__file__).resolve().parents[1] / "reference_results"
    for archive, expected in (("uea", UEA_DATASETS), ("ucr", UCR_DATASETS)):
        with (root / archive / "d32" / "phiAPE_runs.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            observed = {row["dataset"] for row in csv.DictReader(stream)}
        assert observed == set(expected)
        assert len(expected) == (30 if archive == "uea" else 128)


def test_reference_manifest_hash_and_coverage_validation(tmp_path) -> None:
    configurations = []
    for archive in ARCHIVES:
        for dimension in DIMENSIONS:
            directory = tmp_path / archive / f"d{dimension}"
            directory.mkdir(parents=True)
            result_file = directory / "phiAPE_runs.csv"
            result_file.write_text("dataset\nTiny\n", encoding="utf-8")
            metadata_file = directory / "phiAPE_run_metadata.json"
            metadata_file.write_text("{}\n", encoding="utf-8")
            configurations.append(
                {
                    "archive": archive,
                    "dimension": dimension,
                    "datasets": 1,
                    "rows": 1,
                    "results_path": str(result_file.relative_to(tmp_path)),
                    "results_sha256": sha256_file(result_file),
                    "metadata_path": str(metadata_file.relative_to(tmp_path)),
                    "metadata_sha256": sha256_file(metadata_file),
                }
            )
    manifest = {
        "schema_version": 1,
        "methods": list(METHODS),
        "protocol": PROTOCOL.to_dict(),
        "configurations": configurations,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_reference_manifest(tmp_path) == manifest

    (tmp_path / "uea" / "d32" / "phiAPE_runs.csv").write_text(
        "dataset\nChanged\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_reference_manifest(tmp_path)
