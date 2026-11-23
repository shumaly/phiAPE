"""Resumable execution of the locked phiAPE experiments."""

from __future__ import annotations

import json
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path

import aeon
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from . import __version__
from .core import (
    make_tensor_dataset,
    split_train_validation,
    standardize_from_training,
    train_one_run,
)
from .data import dataset_names, load_dataset
from .protocol import ARCHIVES, DIMENSIONS, METHODS, PROTOCOL, RESULT_COLUMNS
from .statistics import validate_runs
from .rotary_attention import ROTARY_METHODS


def configuration_directory(output_root: Path, archive: str, dimension: int) -> Path:
    return output_root / archive / f"d{dimension}"


def results_path(output_root: Path, archive: str, dimension: int) -> Path:
    return configuration_directory(output_root, archive, dimension) / "phiAPE_runs.csv"


def metadata_path(output_root: Path, archive: str, dimension: int) -> Path:
    return configuration_directory(output_root, archive, dimension) / "phiAPE_run_metadata.json"


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_metadata(device: torch.device) -> dict[str, object]:
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "sktime": installed_version("sktime"),
        "aeon": aeon.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def expected_metadata(archive: str, dimension: int, device: torch.device, methods: tuple[str, ...] = METHODS) -> dict[str, object]:
    return {
        "archive": archive,
        "dimension": dimension,
        "methods": list(methods),
        "protocol": PROTOCOL.to_dict(),
        "environment": environment_metadata(device),
    }


def _validate_existing_metadata(path: Path, expected: dict[str, object]) -> None:
    if not path.exists():
        return
    observed = json.loads(path.read_text(encoding="utf-8"))
    for key in ("archive", "dimension", "methods", "protocol", "environment"):
        if observed.get(key) != expected[key]:
            raise ValueError(f"existing output metadata has incompatible {key}: {path}")


def validate_metadata(path: Path, archive: str, dimension: int, methods: tuple[str, ...] = METHODS) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing run metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_static = {
        "archive": archive,
        "dimension": dimension,
        "methods": list(methods),
        "protocol": PROTOCOL.to_dict(),
    }
    for key, expected_value in expected_static.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"run metadata has incompatible {key}: {path}")
    if not isinstance(metadata.get("environment"), dict):
        raise ValueError(f"run metadata is missing environment information: {path}")
    return metadata


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(RESULT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if frame.duplicated(["dataset", "positional_encoding", "repeat"]).any():
        raise ValueError(f"duplicate result keys in {path}")
    return frame.loc[:, list(RESULT_COLUMNS)].copy()


def save_results(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.loc[:, list(RESULT_COLUMNS)].sort_values(
        ["dataset", "positional_encoding", "repeat"], kind="stable"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def run_configuration(
    archive: str,
    dimension: int,
    output_root: Path,
    data_root: Path,
    selected_datasets: tuple[str, ...] | None = None,
    selected_methods: tuple[str, ...] | None = None,
    selected_seeds: tuple[int, ...] | None = None,
    rotary: bool = False,
) -> Path:
    archive = archive.lower()
    if archive not in ARCHIVES:
        raise ValueError(f"archive must be one of {ARCHIVES}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    available_datasets = dataset_names(archive)
    datasets = selected_datasets or available_datasets
    family = ROTARY_METHODS if rotary else METHODS
    if rotary:
        output_root = output_root / "rotary"
    methods = selected_methods or family
    seeds = selected_seeds or PROTOCOL.model_seeds
    unknown_datasets = set(datasets) - set(available_datasets)
    unknown_methods = set(methods) - set(family)
    unknown_seeds = set(seeds) - set(PROTOCOL.model_seeds)
    if unknown_datasets:
        raise ValueError(f"unknown {archive.upper()} datasets: {sorted(unknown_datasets)}")
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    if unknown_seeds:
        raise ValueError(f"seeds outside the published protocol: {sorted(unknown_seeds)}")

    device = choose_device()
    result_file = results_path(output_root, archive, dimension)
    metadata_file = metadata_path(output_root, archive, dimension)
    metadata = expected_metadata(archive, dimension, device, family)
    _validate_existing_metadata(metadata_file, metadata)
    if not metadata_file.exists():
        _write_json(metadata_file, metadata)
    results = load_results(result_file)
    if not results.empty:
        validate_runs(results, archive, dimension, complete=False, methods=family)
    completed = set(
        zip(
            results["dataset"].astype(str),
            results["positional_encoding"].astype(str),
            results["repeat"].astype(int),
        )
    )
    print(f"Archive={archive.upper()} d={dimension} device={device}")
    for dataset in datasets:
        pending = [
            (method, seed)
            for method in methods
            for seed in seeds
            if (dataset, method, PROTOCOL.repeat_for_seed(seed)) not in completed
        ]
        if not pending:
            print(f"{dataset}: already complete for the requested runs")
            continue
        print(f"{dataset}: loading data")
        x_train, y_train, x_test, y_test, sequence_length, original_length, classes = load_dataset(
            archive, dataset, data_root, PROTOCOL.maximum_length
        )
        x_train, x_validation, y_train, y_validation = split_train_validation(x_train, y_train)
        x_train, x_validation, x_test = standardize_from_training(x_train, x_validation, x_test)
        train_data = make_tensor_dataset(x_train, y_train)
        validation_data = make_tensor_dataset(x_validation, y_validation)
        test_data = make_tensor_dataset(x_test, y_test)
        for method, seed in pending:
            repeat = PROTOCOL.repeat_for_seed(seed)
            validation_accuracy, test_loss, test_accuracy, epochs_trained = train_one_run(
                train_data,
                validation_data,
                test_data,
                input_channels=x_train.shape[1],
                sequence_length=sequence_length,
                num_classes=classes,
                method=method,
                seed=seed,
                embedding_size=dimension,
                device=device,
            )
            row = {
                "dataset": dataset,
                "train_cases": len(train_data),
                "test_cases": len(test_data),
                "channels": x_train.shape[1],
                "sequence_length": sequence_length,
                "original_length": original_length,
                "classes": classes,
                "embedding_size": dimension,
                "positional_encoding": method,
                "repeat": repeat,
                "seed": seed,
                "val_accuracy": validation_accuracy,
                "test_loss": test_loss,
                "test_correct": int(round(test_accuracy * len(test_data))),
                "test_accuracy": test_accuracy,
            }
            new_row = pd.DataFrame([row], columns=RESULT_COLUMNS)
            results = new_row if results.empty else pd.concat([results, new_row], ignore_index=True)
            completed.add((dataset, method, repeat))
            save_results(results, result_file)
            print(
                f"{dataset} | {method} | repeat={repeat} seed={seed} "
                f"epochs={epochs_trained} test_accuracy={test_accuracy:.12g}"
            )
    return result_file


def run_all(output_root: Path, data_root: Path, rotary: bool = False) -> None:
    for archive in ARCHIVES:
        for dimension in DIMENSIONS:
            run_configuration(archive, dimension, output_root, data_root, rotary=rotary)
