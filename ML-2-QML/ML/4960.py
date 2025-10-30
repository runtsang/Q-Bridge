"""Hybrid fraud‑detection model that fuses classical photonic layers,
quantum‑kernel embeddings, an autoencoder, and a regression head.

The module is deliberately importable from both the original
`FraudDetection.py` and the new target file, preserving the data‑flow
signature while extending the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

# Import the photonic‑style utilities from the anchor seed.
from FraudDetection import FraudLayerParameters, build_fraud_detection_program
# Import the kernel helper (classical RBF) from the quantum‑kernel pair.
from QuantumKernelMethod import Kernel
# Import the autoencoder factory from the autoencoder pair.
from Autoencoder import Autoencoder

class FraudDetectionHybrid(nn.Module):
    """
    A hybrid model that stacks:
    1. Classical photonic‑style layers (`build_fraud_detection_program`).
    2. A quantum‑kernel embedding (`Kernel`).
    3. An autoencoder that compresses the kernel output.
    4. A regression head that mirrors the EstimatorQNN architecture.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        kernel_gamma: float = 1.0,
        autoencoder_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        # Classical photonic layers
        self.fraud_program = build_fraud_detection_program(input_params, layers)

        # Quantum‑kernel embedding (RBF or TorchQuantum)
        self.kernel = Kernel(gamma=kernel_gamma)

        # Autoencoder for dimensionality reduction
        self.autoencoder = Autoencoder(**(autoencoder_cfg or {}))

        # Regression head (EstimatorQNN‑style)
        self.regressor = nn.Sequential(
            nn.Linear(self.autoencoder.decoder[-1].out_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1️⃣ Classical photonic layers
        x = self.fraud_program(x)

        # 2️⃣ Quantum‑kernel embedding (self‑similarity)
        kernel_val = self.kernel(x, x)  # shape: (batch, 1)
        x = kernel_val

        # 3️⃣ Autoencoder compression
        z = self.autoencoder.encode(x)

        # 4️⃣ Regression output
        return self.regressor(z)

    def fit(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Simple training loop that returns the loss history."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred = self.forward(data)
            loss = loss_fn(pred.squeeze(), targets)
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        return history

__all__ = ["FraudDetectionHybrid"]
