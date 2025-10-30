"""Quantum fraud detection model using Pennylane with variational circuits and hardware backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used as variational parameters)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    wires: Sequence[int], params: FraudLayerParameters, *, clip: bool
) -> None:
    """Apply a photonic‑style layer using qubit gates."""
    # Encode phases as Z rotations
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])
    # Squeezing -> Y rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RY(_clip(r, 5.0) if clip else r, wires=wires[i])
    # Entanglement mimicking a beamsplitter
    qml.CNOT(wires[0], wires[1])
    # Displacement -> X rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5.0) if clip else r, wires=wires[i])
    # Kerr -> Z rotations
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0) if clip else k, wires=wires[i])


def build_fraud_detection_qnode(
    input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]
) -> qml.QNode:
    """Create a Pennylane QNode that mirrors the photonic fraud detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # Angle‑encoding of the two‑dimensional input
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        # Measure expectation values of PauliZ on each wire
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit


class FraudDetectorQ(nn.Module):
    """Hybrid quantum‑classical fraud detector with a variational circuit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.circuit = build_fraud_detection_qnode(input_params, layers)
        # Linear read‑out mapping from two expectation values to a single logit
        self.readout = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expectation values from the quantum circuit
        exps = self.circuit(x)
        logits = self.readout(exps)
        return torch.sigmoid(logits)

    def train_model(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float = 1e-3,
        device: str = "cpu",
        patience: int = 5,
    ) -> None:
        """Training loop using Pennylane’s autograd with Adam optimizer."""
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                optimizer.zero_grad()
                preds = self.forward(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(data_loader.dataset)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
        """Return average accuracy on a dataset."""
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                preds = self.forward(xb).squeeze() > 0.5
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total


__all__ = ["FraudLayerParameters", "build_fraud_detection_qnode", "FraudDetectorQ"]
