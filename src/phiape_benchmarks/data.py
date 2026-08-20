"""Archive-specific loaders with shared published preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .protocol import UCR_DATASETS, UEA_DATASETS


DatasetBundle = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]


def resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) == target_length:
        return values.astype(np.float32, copy=False)
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(values) == 1:
        return np.full(target_length, float(values[0]), dtype=np.float32)
    old_positions = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(new_positions, old_positions, values).astype(np.float32)


def _series_to_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if np.isnan(array).any():
        array = (
            pd.Series(array)
            .interpolate(limit_direction="both")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    return array


def _maximum_nested_length(collection: pd.DataFrame) -> int:
    maximum = max(
        (len(_series_to_array(row[column])) for _, row in collection.iterrows() for column in collection.columns),
        default=0,
    )
    if maximum == 0:
        raise ValueError("dataset contains no time-series values")
    return maximum


def _nested_to_numpy(collection: pd.DataFrame, target_length: int) -> np.ndarray:
    samples = []
    for _, row in collection.iterrows():
        channels = [resample_1d(_series_to_array(row[column]), target_length) for column in collection.columns]
        samples.append(np.stack(channels, axis=0))
    return np.stack(samples, axis=0).astype(np.float32)


def _classification_to_numpy(collection: object) -> np.ndarray:
    array = np.asarray(collection, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, np.newaxis, :]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D time-series array, got {array.shape}")
    if np.isnan(array).any():
        array = np.nan_to_num(array, copy=False)
    return array


def _resample_collection(collection: np.ndarray, target_length: int) -> np.ndarray:
    if collection.shape[2] == target_length:
        return collection.astype(np.float32, copy=False)
    return np.stack(
        [
            np.stack([resample_1d(sample[channel], target_length) for channel in range(collection.shape[1])])
            for sample in collection
        ]
    ).astype(np.float32)


def _encode_labels(y_train: object, y_test: object) -> tuple[np.ndarray, np.ndarray, int]:
    encoder = LabelEncoder()
    train = encoder.fit_transform(y_train).astype(np.int64)
    test = encoder.transform(y_test).astype(np.int64)
    return train, test, len(encoder.classes_)


def load_uea_dataset(dataset: str, data_directory: Path, maximum_length: int) -> DatasetBundle:
    if dataset not in UEA_DATASETS:
        raise ValueError(f"unknown published UEA dataset: {dataset}")
    from sktime.datasets import load_UCR_UEA_dataset

    data_directory.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_UCR_UEA_dataset(
        dataset, split="train", return_X_y=True, extract_path=str(data_directory)
    )
    test_x, test_y = load_UCR_UEA_dataset(
        dataset, split="test", return_X_y=True, extract_path=str(data_directory)
    )
    original_length = max(_maximum_nested_length(train_x), _maximum_nested_length(test_x))
    target_length = min(original_length, maximum_length)
    if target_length < 2:
        raise ValueError(f"{dataset} target length is too short: {target_length}")
    encoded_train, encoded_test, classes = _encode_labels(train_y, test_y)
    return (
        _nested_to_numpy(train_x, target_length),
        encoded_train,
        _nested_to_numpy(test_x, target_length),
        encoded_test,
        target_length,
        original_length,
        classes,
    )


def load_ucr_dataset(dataset: str, data_directory: Path, maximum_length: int) -> DatasetBundle:
    if dataset not in UCR_DATASETS:
        raise ValueError(f"unknown published UCR dataset: {dataset}")
    from aeon.datasets import load_classification

    data_directory.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_classification(dataset, split="train", extract_path=str(data_directory))
    test_x, test_y = load_classification(dataset, split="test", extract_path=str(data_directory))
    train_array = _classification_to_numpy(train_x)
    test_array = _classification_to_numpy(test_x)
    original_length = max(train_array.shape[2], test_array.shape[2])
    target_length = min(original_length, maximum_length)
    if target_length < 2:
        raise ValueError(f"{dataset} target length is too short: {target_length}")
    encoded_train, encoded_test, classes = _encode_labels(train_y, test_y)
    return (
        _resample_collection(train_array, target_length),
        encoded_train,
        _resample_collection(test_array, target_length),
        encoded_test,
        target_length,
        original_length,
        classes,
    )


def dataset_names(archive: str) -> tuple[str, ...]:
    if archive == "uea":
        return UEA_DATASETS
    if archive == "ucr":
        return UCR_DATASETS
    raise ValueError(f"archive must be uea or ucr; got {archive}")


def load_dataset(archive: str, dataset: str, data_root: Path, maximum_length: int) -> DatasetBundle:
    directory = data_root / archive
    if archive == "uea":
        return load_uea_dataset(dataset, directory, maximum_length)
    if archive == "ucr":
        return load_ucr_dataset(dataset, directory, maximum_length)
    raise ValueError(f"archive must be uea or ucr; got {archive}")
