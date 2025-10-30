"""Enhanced fraud detection pipeline with classical neural network and training utilities.

The module extends the original lightweight model by adding:

* Parameter validation and automatic clipping.
* A convenient ``FraudLayerParameters.to_tensor`` helper for conversion to torch tensors.
* Optional dropout and batch‑normalisation layers.
* A simple end‑to‑end training routine and evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

    def __post_init__(self) -> None:
        """Clip all continuous parameters to the range [−5, 5] (except kerr to [−1, 1])."""
        clip = lambda v: max(-5.0, min(5.0, v))
        self.bs_theta, self.bs_phi = clip(self.bs_theta), clip(self.bs_phi)
        self.phases = tuple(clip(p) for p in self.phases)
        self.squeeze_r = tuple(clip(r) for r in self.squeeze_r)
        self.squeeze_phi = tuple(clip(p) for p in self.squeeze_phi)
        self.displacement_r = tuple(clip(r) for r in self.displacement_r)
        self.displacement_phi = tuple(clip(p) for p in self.displacement_phi)
        self.kerr = tuple(max(-1.0, min(1.0, k)) for k in self.kerr)

    def to_tensor(self) -> dict[str, torch.Tensor]:
        """Return a dictionary of torch tensors for all parameters."""
        return {
            "bs_theta": torch.tensor(self.bs_theta, dtype=torch.float32),
            "bs_phi": torch.tensor(self.bs_phi, dtype=torch.float32),
            "phases": torch.tensor(self.phases, dtype=torch.float32),
            "squeeze_r": torch.tensor(self.squeeze_r, dtype=torch.float32),
            "squeeze_phi": torch.tensor(self.squeeze_phi, dtype=torch.float32),
            "displacement_r": torch.tensor(self.displacement_r, dtype=torch.float32),
            "displacement_phi": torch.tensor(self.displacement_phi, dtype=torch.float32),
            "kerr": torch.tensor(self.kerr, dtype=torch.float32),
        }

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Instantiate a single network block from ``params``."""
    # Build a linear layer with the first two parameters as weights,
    # and the remaining two as bias terms.
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout: Optional[float] = None,
    batchnorm: bool = False,
) -> nn.Sequential:
    """
    Create a sequential PyTorch model mirroring the layered structure.

    Parameters
    ----------
    input_params:
        Parameters of the first (un‑clipped) layer.
    layers:
        Subsequent layers, each clipped to sensible bounds.
    dropout:
        Optional dropout probability applied after every block.
    batchnorm:
        Whether to insert a batch‑norm after each block.
    """
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]

    for layer in layers:
        block = _layer_from_params(layer, clip=True)
        if batchnorm:
            block = nn.Sequential(block, nn.BatchNorm1d(2))
        if dropout is not None:
            block = nn.Sequential(block, nn.Dropout(dropout))
        modules.append(block)

    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

def train_fraud_detector(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """Simple training loop for a fraud‑detection model."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                val_loss += criterion(logits, y.float()).item()
                preds = logits > 0
                correct += preds.eq(y.byte()).sum().item()
                total += y.size(0)
            val_loss /= len(val_loader)
            val_acc = correct / total
        print(f"Epoch {epoch:02d} | Val loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    return model

def evaluate_fraud_detector(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Return (average loss, accuracy) on the supplied data."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            total_loss += criterion(logits, y.float()).item()
            preds = logits > 0
            correct += preds.eq(y.byte()).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "train_fraud_detector",
    "evaluate_fraud_detector",
]
