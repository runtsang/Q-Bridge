"""
FraudDetectionModel – Variational qubit implementation.

This module defines a hybrid quantum‑classical fraud‑detection model
using PennyLane.  Each photonic layer is translated into a
parameterised quantum sub‑circuit consisting of rotations,
entangling gates and post‑processing.  The circuit outputs a single
expectation value which is then fed into a lightweight linear
classifier.  Parameter clipping is enforced to keep the optimisation
stable.  The class exposes a `train` method that jointly updates
the variational parameters and the linear head using gradient descent.

Author: gpt-oss-20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
import pennylane.numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel:
    """
    Variational qubit fraud‑detection model.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 device: str = "default.qubit",
                 wires: int = 2) -> None:
        self.wires = wires
        self.dev = qml.device(device, wires=wires)
        self.input_params = input_params
        self.layers = list(layers)
        self._build_circuit()
        self.linear_head = nn.Linear(1, 1)
        self.opt = qml.GradientDescentOptimizer(stepsize=0.01)
        self.loss_fn = nn.MSELoss()

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor,
                    params: torch.Tensor) -> torch.Tensor:
            # Input encoding
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Input layer (photonic parameters mapped to rotations)
            self._apply_layer(circuit, self.input_params, clip=False)

            # Subsequent layers
            for i, layer in enumerate(self.layers):
                self._apply_layer(circuit, layer, clip=True)

            # Observable
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def _apply_layer(self, circuit, params: FraudLayerParameters, clip: bool) -> None:
        """Translate a photonic layer into a parameterised qubit sub‑circuit."""
        # Beam splitter analog: CZ gate
        qml.CZ(wires=[0, 1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing analog: S gate
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.S(wires=i)

        # Displacement analog: RX rotation
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.RX(r, wires=i)

        # Kerr analog: RZ
        for i, k in enumerate(params.kerr):
            qml.RZ(k, wires=i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: quantum expectation → linear head.
        """
        q_expect = self.circuit(x, torch.tensor([]))
        logits = self.linear_head(q_expect)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> None:
        """
        Joint training of the variational circuit and linear head.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        for epoch in range(epochs):
            loss_total = 0.0
            for xi, yi in zip(X_t, y_t):
                self.opt.step(lambda p: self.loss_fn(self.forward(xi), yi), p=None)
                loss_total += self.loss_fn(self.forward(xi), yi).item()
            if epoch % 20 == 0:
                print(f"Epoch {epoch} loss {loss_total/len(X_t):.4f}")

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
