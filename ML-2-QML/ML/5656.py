"""Enhanced classical fraud detection model with two‑class head and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
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
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure with a 2‑class head."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 2))
    modules.append(nn.LogSoftmax(dim=-1))
    return nn.Sequential(*modules)

class FraudDetectionModel(nn.Module):
    """Wrapper around the sequential program to expose a single‑step forward."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.net = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def temperature_scale(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    return logits / temperature

def calibrate_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    init_temp: float = 1.0,
    lr: float = 0.01,
    epochs: int = 200,
) -> torch.Tensor:
    """Find optimal temperature that maximises validation log‑likelihood."""
    temperature = torch.tensor(init_temp, requires_grad=True, device=logits.device)
    optimizer = optim.Adam([temperature], lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.nll_loss(temperature_scale(logits, temperature), targets)
        loss.backward()
        optimizer.step()
        temperature.detach_()
        temperature.clamp_(min=1e-3)
    return temperature

def train_model(
    model: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    patience: int,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, torch.Tensor]:
    """Train with early stopping and return the best model."""
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model, torch.tensor(best_val_loss, device=device)

def run_experiment(
    input_params: FraudLayerParameters,
    layer_params_list: List[FraudLayerParameters],
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    test_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    hyperparams: dict,
) -> dict:
    """Execute a full training‑calibration pipeline and return metrics."""
    model = FraudDetectionModel(input_params, layer_params_list)
    model, val_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=hyperparams.get("epochs", 30),
        patience=hyperparams.get("patience", 5),
        lr=hyperparams.get("lr", 1e-3),
    )
    # Calibration on validation logits
    all_logits = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            all_logits.append(logits)
            all_targets.append(y)
    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    temperature = calibrate_temperature(logits_cat, targets_cat)
    # Apply calibrated temperature to test set
    test_logits = []
    test_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            test_logits.append(temperature_scale(logits, temperature))
            test_labels.append(y)
    test_logits_cat = torch.cat(test_logits)
    test_labels_cat = torch.cat(test_labels)
    test_pred = torch.argmax(test_logits_cat, dim=-1)
    accuracy = (test_pred == test_labels_cat).float().mean()
    return {
        "val_loss": val_loss.item(),
        "temperature": temperature.item(),
        "test_accuracy": accuracy.item(),
    }

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
    "build_fraud_detection_program",
    "train_model",
    "calibrate_temperature",
    "run_experiment",
]
