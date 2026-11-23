from __future__ import annotations

import math

import numpy as np
import torch

from phiape_benchmarks.core import (
    TSTClassifier,
    frequency_phiape,
    positional_encoding,
    split_train_validation,
    standardize_from_training,
)
from phiape_benchmarks.protocol import PROTOCOL


def test_seed_repeat_mapping() -> None:
    assert PROTOCOL.model_seeds == tuple(range(2026, 2042))
    assert PROTOCOL.repeat_for_seed(2026) == 1
    assert PROTOCOL.repeat_for_seed(2041) == 16


def test_phiape_frequency_range_and_encoding() -> None:
    device = torch.device("cpu")
    frequencies = frequency_phiape(25, 32, device)
    minimum = math.pi / 24
    maximum = minimum + (32 / 4) * minimum
    assert frequencies.shape == (16,)
    assert torch.isclose(frequencies.min(), torch.tensor(minimum), atol=1e-7)
    assert torch.isclose(frequencies.max(), torch.tensor(maximum), atol=1e-7)
    encoding = positional_encoding(25, 32, device, frequency_phiape)
    assert encoding.shape == (1, 25, 32)
    assert torch.equal(encoding[0, 0, 0::2], torch.zeros(16))
    assert torch.equal(encoding[0, 0, 1::2], torch.ones(16))


def test_split_is_deterministic_and_standardization_uses_training_only() -> None:
    features = np.arange(48, dtype=np.float32).reshape(12, 2, 2)
    labels = np.repeat(np.arange(3), 4)
    first = split_train_validation(features, labels)
    second = split_train_validation(features, labels)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    train = np.array([[[1.0, 3.0]], [[5.0, 7.0]]], dtype=np.float32)
    validation = np.array([[[9.0, 11.0]]], dtype=np.float32)
    test = np.array([[[13.0, 15.0]]], dtype=np.float32)
    scaled_train, scaled_validation, scaled_test = standardize_from_training(train, validation, test)
    assert np.isclose(scaled_train.mean(), 0.0)
    assert np.isclose(scaled_train.std(), 1.0)
    assert scaled_validation.mean() > scaled_train.mean()
    assert scaled_test.mean() > scaled_validation.mean()


def test_model_forward_for_all_methods() -> None:
    batch = torch.randn(3, 12, 2)
    for method in ("No-APE", "Learned APE", "Sin-APE", "tAPE", "phiAPE", "RoPE", "phiRoPE"):
        model = TSTClassifier(2, 12, 4, method, d_model=32)
        model.eval()
        output = model(batch)
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()
