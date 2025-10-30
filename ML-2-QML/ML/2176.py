"""Robust classical classifier with enhanced training utilities."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam


class QuantumClassifierModel(nn.Module):
    """Feed‑forward network mirroring the quantum helper interface.

    Features:
    * configurable hidden size, dropout and batch‑norm
    * ``train_step`` and ``test_step`` helpers
    * ``export_onnx`` for model deployment
    * metadata attributes (`encoding`, ``weight_sizes``, ``observables``)
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_size: int = 64,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

        # expose metadata for compatibility
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables = list(range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: Adam,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> float:
        x, y = batch
        optimizer.zero_grad()
        logits = self(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        with torch.no_grad():
            logits = self(x)
            preds = logits.argmax(dim=1)
        return preds, y

    def export_onnx(self, path: str, sample_input: torch.Tensor) -> None:
        """Export the network to ONNX format."""
        torch.onnx.export(
            self,
            sample_input,
            path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=11,
        )


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_size: int = 64,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> Tuple[QuantumClassifierModel, Iterable[int], Iterable[int], list[int]]:
    """Compatibility wrapper that returns the classical model and metadata."""
    model = QuantumClassifierModel(
        num_features,
        depth,
        hidden_size=hidden_size,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    return model, model.encoding, model.weight_sizes, model.observables
