"""Improved classical fraud detection model with dropout, batch‑norm, and TorchScript export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _make_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2, bias=True)
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


class FraudDetectionModel(nn.Module):
    """
    Classical fraud‑detection model mirroring the photonic circuit.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (unclipped).
    layers : Iterable[FraudLayerParameters]
        Subsequent layers (clipped to prevent exploding gradients).
    dropout : float, optional
        Dropout probability applied after each layer.
    batch_norm : bool, optional
        If True, inserts a BatchNorm1d after each layer.
    device : str | torch.device, optional
        Target device for the model.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [_make_layer(input_params, clip=False)]

        for layer_params in layers:
            modules.append(_make_layer(layer_params, clip=True))
            if batch_norm:
                modules.append(nn.BatchNorm1d(2))
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_one_epoch(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
    ) -> float:
        self.train()
        epoch_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = self.forward(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        return epoch_loss / len(dataloader.dataset)

    def evaluate(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        device: str | torch.device = "cpu",
    ) -> float:
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self.forward(xb)
                total_loss += loss_fn(preds, yb).item() * xb.size(0)
        return total_loss / len(dataloader.dataset)

    def export_script(self, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """
        Export the model to TorchScript for deployment.

        Parameters
        ----------
        example_input : torch.Tensor
            A sample input tensor to trace the model.
        """
        example_input = example_input.to(next(self.parameters()).device)
        return torch.jit.trace(self.model, example_input)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
