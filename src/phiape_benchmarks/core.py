"""Shared phiAPE positional encoding, TST model, and training implementation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .protocol import PROTOCOL
from .rotary_attention import ROTARY_METHODS, RotaryMultiheadAttention


FrequencyBuilder = Callable[[int, int, torch.device], torch.Tensor]
PUBLISHED_RANGE_DIVISOR = 4.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def frequency_sin_ape(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    return 10000.0 ** (-indices / d_model)


def frequency_tape(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    return frequency_sin_ape(length, d_model, device) * (d_model / length)


def frequency_phiape(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    raw = 10000.0 ** (-indices / d_model)
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    minimum = math.pi / max(length - 1, 1)
    maximum = minimum + (d_model / PUBLISHED_RANGE_DIVISOR) * minimum
    return minimum + normalized * (maximum - minimum)


FREQUENCY_BUILDERS: dict[str, FrequencyBuilder] = {
    "Sin-APE": frequency_sin_ape,
    "tAPE": frequency_tape,
    "phiAPE": frequency_phiape,
}


def positional_encoding(
    length: int,
    d_model: int,
    device: torch.device,
    frequency_builder: FrequencyBuilder,
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    angles = positions * frequency_builder(length, d_model, device).unsqueeze(0)
    encoding = torch.zeros(length, d_model, device=device)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding.unsqueeze(0)


def split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_fraction: float = PROTOCOL.validation_fraction,
    split_seed: int = PROTOCOL.split_seed,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y_train) < 3 or validation_fraction <= 0:
        return x_train, y_train, x_train, y_train
    stratify = y_train if np.min(np.bincount(y_train)) >= 2 else None
    try:
        return train_test_split(
            x_train,
            y_train,
            test_size=validation_fraction,
            random_state=split_seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            x_train,
            y_train,
            test_size=validation_fraction,
            random_state=split_seed,
            stratify=None,
        )


def standardize_from_training(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    standard_deviation = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (
        (x_train - mean) / standard_deviation,
        (x_validation - mean) / standard_deviation,
        (x_test - mean) / standard_deviation,
    )


def make_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    features = torch.from_numpy(np.transpose(x, (0, 2, 1)).astype(np.float32, copy=False))
    labels = torch.from_numpy(y.astype(np.int64, copy=False))
    return TensorDataset(features, labels)


class TSTBatchNormEncoderLayer(nn.Module):
    """Transformer encoder layer used by the published TST model."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = PROTOCOL.attention_heads,
        dim_feedforward: int = PROTOCOL.feedforward_dimension,
        dropout: float = PROTOCOL.dropout,
        rotary_method: str | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = (
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
            if rotary_method is None else
            RotaryMultiheadAttention(d_model, num_heads, dropout=dropout, method=rotary_method)
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        source: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attention = self.self_attn(
            source,
            source,
            source,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        source = source + self.dropout1(attention)
        source = self.norm1(source.permute(1, 2, 0)).permute(2, 0, 1)
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(source))))
        source = source + self.dropout2(feedforward)
        return self.norm2(source.permute(1, 2, 0)).permute(2, 0, 1)


class TSTClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        num_classes: int,
        method: str,
        d_model: int,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.method = method
        self.input_projection = nn.Linear(input_channels, d_model)
        if method == "Learned APE":
            self.learnable_encoding = nn.Parameter(torch.empty(sequence_length, d_model))
            nn.init.uniform_(self.learnable_encoding, -0.02, 0.02)
        else:
            self.learnable_encoding = None
        self.positional_dropout = nn.Dropout(PROTOCOL.dropout)
        layer = TSTBatchNormEncoderLayer(d_model=d_model, rotary_method=method if method in ROTARY_METHODS else None)
        self.encoder = nn.TransformerEncoder(layer, num_layers=PROTOCOL.encoder_layers)
        self.activation = nn.GELU()
        self.output_dropout = nn.Dropout(PROTOCOL.dropout)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_masks = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        hidden = self.input_projection(x.permute(1, 0, 2))
        if PROTOCOL.scale_embedding:
            hidden = hidden * math.sqrt(self.d_model)
        if self.method == "Learned APE":
            hidden = hidden + self.learnable_encoding.unsqueeze(1)
        elif self.method not in ("No-APE", *ROTARY_METHODS):
            encoding = positional_encoding(
                self.sequence_length,
                self.d_model,
                x.device,
                FREQUENCY_BUILDERS[self.method],
            )
            hidden = hidden + encoding.squeeze(0).unsqueeze(1)
        hidden = self.positional_dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~padding_masks)
        hidden = self.activation(hidden)
        hidden = self.output_dropout(hidden.permute(1, 0, 2)).mean(dim=1)
        return self.output_layer(hidden)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(label_smoothing=PROTOCOL.validation_label_smoothing)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            total_loss += float(loss_function(logits, labels).item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_run(
    train_data: TensorDataset,
    validation_data: TensorDataset,
    test_data: TensorDataset,
    input_channels: int,
    sequence_length: int,
    num_classes: int,
    method: str,
    seed: int,
    embedding_size: int,
    device: torch.device,
) -> tuple[float, float, float, int]:
    set_seed(seed)
    train_loader = DataLoader(train_data, batch_size=PROTOCOL.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=PROTOCOL.batch_size)
    test_loader = DataLoader(test_data, batch_size=PROTOCOL.batch_size)
    model = TSTClassifier(
        input_channels=input_channels,
        sequence_length=sequence_length,
        num_classes=num_classes,
        method=method,
        d_model=embedding_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PROTOCOL.learning_rate,
        weight_decay=PROTOCOL.weight_decay,
    )
    training_loss = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    for epoch in range(1, PROTOCOL.epochs + 1):
        epochs_trained = epoch
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL.gradient_clip_norm)
            optimizer.step()
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PROTOCOL.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    return best_validation_accuracy, test_loss, test_accuracy, epochs_trained
