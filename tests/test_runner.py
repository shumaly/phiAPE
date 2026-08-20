from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phiape_benchmarks import data
from phiape_benchmarks import runner
from phiape_benchmarks.cli import default_data_root


def test_default_dataset_root_and_archive_subdirectories(tmp_path, monkeypatch) -> None:
    assert default_data_root() == Path("datasets")
    expected = (
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        2,
        2,
        1,
    )
    observed: dict[str, Path] = {}

    def fake_loader(dataset, directory, maximum_length):
        observed["directory"] = directory
        return expected

    monkeypatch.setattr(data, "load_uea_dataset", fake_loader)
    assert data.load_dataset("uea", "ExampleUEA", tmp_path / "datasets", 512) is expected
    assert observed["directory"] == tmp_path / "datasets" / "uea"
    monkeypatch.setattr(data, "load_ucr_dataset", fake_loader)
    assert data.load_dataset("ucr", "Coffee", tmp_path / "datasets", 512) is expected
    assert observed["directory"] == tmp_path / "datasets" / "ucr"


def test_resumable_single_run_and_metadata_guard(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    train_x = np.arange(48, dtype=np.float32).reshape(12, 2, 2)
    train_y = np.repeat(np.arange(3), 4)
    test_x = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    test_y = np.repeat(np.arange(3), 2)

    monkeypatch.setattr(runner, "dataset_names", lambda archive: ("Tiny",))
    monkeypatch.setattr("phiape_benchmarks.statistics.dataset_names", lambda archive: ("Tiny",))
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda archive, dataset, data_root, maximum_length: (
            train_x,
            train_y,
            test_x,
            test_y,
            2,
            2,
            3,
        ),
    )

    def fake_train(*args, method, seed, **kwargs):
        calls.append((method, seed))
        return 0.5, 0.75, 0.5, 3

    monkeypatch.setattr(runner, "train_one_run", fake_train)
    output_root = tmp_path / "outputs"
    arguments = dict(
        archive="uea",
        dimension=32,
        output_root=output_root,
        data_root=tmp_path / "data",
        selected_datasets=("Tiny",),
        selected_methods=("phiAPE",),
        selected_seeds=(2026,),
    )
    path = runner.run_configuration(**arguments)
    assert calls == [("phiAPE", 2026)]
    runner.run_configuration(**arguments)
    assert calls == [("phiAPE", 2026)]
    frame = runner.load_results(path)
    assert len(frame) == 1
    assert frame.iloc[0]["repeat"] == 1

    metadata_file = runner.metadata_path(output_root, "uea", 32)
    metadata = json.loads(metadata_file.read_text())
    metadata["dimension"] = 128
    metadata_file.write_text(json.dumps(metadata))
    try:
        runner.run_configuration(**arguments)
    except ValueError as exc:
        assert "incompatible dimension" in str(exc)
    else:
        raise AssertionError("incompatible metadata was accepted")
