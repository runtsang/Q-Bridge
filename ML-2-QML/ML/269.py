"""QuantumHybridClassifier – classical backbone with optional calibration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumHybridClassifier", "CalibrationLayer"]

class CalibrationLayer(nn.Module):
    """Calibrates logits to a smoother probability distribution via a learnable scale."""
    def __init__(self, in_features: int, out_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        # Apply learned log‑scale to control sharpness
        logits = logits / (1 + torch.exp(-self.scale))
        return F.log_softmax(logits, dim=-1)

class QuantumHybridClassifier(nn.Module):
    """Hybrid model that couples a classical MLP with a quantum expectation layer."""
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list[int],
                 output_dim: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.mlp = self._build_mlp()
        self.calibration = CalibrationLayer(self.mlp.out_features, self.output_dim)

    def _build_mlp(self) -> nn.Sequential:
        layers = []
        in_f = self.input_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(in_f, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            in_f = hidden
        layers.append(nn.Linear(in_f, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        return self.calibration(logits)

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module) -> float:
        self.train()
        optimizer.zero_grad()
        logits = self(batch["input"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        return loss.item()
