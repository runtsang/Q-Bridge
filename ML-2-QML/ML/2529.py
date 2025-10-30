"""Hybrid classical autoencoder with fraud detection.

This module builds on the simple fully‑connected autoencoder from
Autoencoder.py and augments it with a fraud‑detection sub‑network
parameterised by FraudLayerParameters.  The fraud detector operates
on the latent representation produced by the encoder.  The class
provides a convenient training routine that first optimises the
autoencoder and then fine‑tunes the fraud detector.

The design follows the pattern of the original seed but adds:
* parameter clipping for stability (inspired by FraudDetection.py)
* an explicit `classify` method that returns a probability of fraud
* a factory that accepts a list of FraudLayerParameters to build the
  detection network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the original autoencoder components
from Autoencoder import AutoencoderNet, AutoencoderConfig, Autoencoder, train_autoencoder


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the fraud detector."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single fraud‑detection layer from parameters."""
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


def build_fraud_detection_network(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential fraud‑detection network."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridAutoEncoderFraudDetector(nn.Module):
    """Combines a classical autoencoder with a fraud‑detection head."""
    def __init__(
        self,
        ae_config: AutoencoderConfig,
        fraud_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(ae_config)
        # The fraud detector expects 2‑dimensional latent vectors
        # (the first two dimensions of the latent space are used).
        self.fraud_detector = build_fraud_detection_network(
            next(iter(fraud_params)),
            fraud_params,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(latents)

    def classify(self, latents: torch.Tensor) -> torch.Tensor:
        """Return a fraud probability in [0,1]."""
        logits = self.fraud_detector(latents)
        return torch.sigmoid(logits).squeeze(-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return reconstruction and fraud probability."""
        latents = self.encode(inputs)
        recon = self.decode(latents)
        fraud = self.classify(latents)
        return recon, fraud

    def train_autoencoder(self, data: torch.Tensor, *, epochs: int = 100,
                          batch_size: int = 64, lr: float = 1e-3,
                          weight_decay: float = 0.0, device: torch.device | None = None
                          ) -> List[float]:
        """Train only the autoencoder part."""
        return train_autoencoder(
            self.autoencoder,
            data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

    def train_fraud_detector(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> List[float]:
        """Fine‑tune the fraud detector on latent representations."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.fraud_detector.to(device)
        dataset = TensorDataset(_as_tensor(data), _as_tensor(labels))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.fraud_detector.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                latents = self.autoencoder.encode(batch_x)
                logits = self.fraud_detector(latents)
                loss = loss_fn(logits.squeeze(-1), batch_y.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)
