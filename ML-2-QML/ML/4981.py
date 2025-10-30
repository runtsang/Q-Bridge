"""Hybrid fraud detection model combining photonic-inspired layers, convolutional filtering,
autoencoding, and a regression head.

This module defines :class:`FraudDetectorHybrid`, a PyTorch ``nn.Module`` that
mirrors the structure of the original fraud‑detection example but augments it
with a learnable linear mapping from a convolutional filter, a fully‑connected
auto‑encoder for dimensionality reduction, and a small feed‑forward estimator.
The construction of the linear mapping uses the same parameter schema that
was used in the photonic reference, allowing easy comparison between the
classical and quantum back‑ends.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# Import reusable blocks from the reference modules
from Conv import Conv
from Autoencoder import Autoencoder, AutoencoderConfig
from EstimatorQNN import EstimatorQNN


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Linear:
    """Build a linear layer whose weights and bias are derived from
    the photonic parameters.  ``clip`` enforces the same bounds used in
    the quantum implementation.
    """
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
    return linear


class FraudDetectorHybrid(nn.Module):
    """Hybrid classical fraud detector.

    The network consists of:
        * A 2×2 convolutional filter (classical analogue of a quanvolution)
        * A linear mapping derived from :class:`FraudLayerParameters`
        * A fully‑connected auto‑encoder for feature compression
        * A lightweight estimator head
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        autoencoder_cfg: AutoencoderConfig | None = None,
    ) -> None:
        super().__init__()
        # Linear mapping from photonic parameters
        self.linear = _layer_from_params(fraud_params, clip=False)
        # Classical convolutional filter
        self.conv = Conv(kernel_size=conv_kernel, threshold=conv_threshold)
        # Auto‑encoder
        self.autoencoder = Autoencoder(
            input_dim=2,
            latent_dim=autoencoder_cfg.latent_dim if autoencoder_cfg else 32,
            hidden_dims=autoencoder_cfg.hidden_dims if autoencoder_cfg else (128, 64),
            dropout=autoencoder_cfg.dropout if autoencoder_cfg else 0.1,
        )
        # Estimator head
        self.estimator = EstimatorQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. Convolutional feature extraction
        conv_out = self.conv.run(x)  # scalar in original, we reshape to 2‑dim vector
        conv_vec = torch.tensor([conv_out, conv_out], dtype=torch.float32, device=x.device)

        # 2. Linear mapping
        lin_out = self.linear(conv_vec)

        # 3. Auto‑encoder compression
        latent = self.autoencoder.encode(lin_out)

        # 4. Estimation
        pred = self.estimator(latent)
        return pred

    # ------------------------------------------------------------------
    # Helpers for training
    # ------------------------------------------------------------------
    def train_autoencoder(self, data: torch.Tensor, epochs: int = 50, lr: float = 1e-3) -> list[float]:
        """Train only the auto‑encoder part."""
        device = self.autoencoder.encoder.weight.device
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            recon = self.autoencoder(data)
            loss = loss_fn(recon, data)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    def train_estimator(self, data: torch.Tensor, targets: torch.Tensor, epochs: int = 50, lr: float = 1e-3) -> list[float]:
        """Train only the estimator head."""
        device = self.estimator.weight.device
        self.estimator.train()
        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            preds = self.estimator(data)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history


__all__ = ["FraudLayerParameters", "FraudDetectorHybrid", "_layer_from_params"]
