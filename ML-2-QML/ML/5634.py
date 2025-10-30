"""Enhanced classical fraud‑detection model with dropout and early‑stopping."""
from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List, Dict, Tuple

from dataclasses import dataclass

@dataclass
class FraudLayerParams:
    """Parameters for a single fully‑connected layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudDetectionEnhanced:
    """Shared class for classical and quantum fraud‑detection experiments."""

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    @staticmethod
    def _layer_from_params(params: FraudLayerParams, *, clip: bool) -> nn.Module:
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

    @classmethod
    def build_model(
        cls,
        input_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
        dropout_prob: float = 0.2,
    ) -> nn.Sequential:
        """Construct a dropout‑augmented sequential model."""
        modules: List[nn.Module] = [cls._layer_from_params(input_params, clip=False)]
        modules.extend(cls._layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Dropout(dropout_prob))
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    @staticmethod
    def train(
        model: nn.Module,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        patience: int = 5,
        device: torch.device | str = "cpu",
    ) -> nn.Module:
        """Simple training loop with patience‑based early stopping."""
        model.to(device)
        best_loss = float("inf")
        epochs_no_improve = 0
        best_state_dict = None

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch in dataloader:
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(dataloader.dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                best_state_dict = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        return model

__all__ = ["FraudLayerParams", "FraudDetectionEnhanced"]
