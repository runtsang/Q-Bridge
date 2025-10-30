"""FraudDetectionAdvanced: quantum implementation using PennyLane.

This module implements a hybrid‑variational circuit that mirrors the
classical layer parameters.  The circuit is defined as a Pennylane QNode
and can be executed on any device that supports the Pennylane interface,
including 'default.qubit', Qiskit backends, or real hardware.
The class FraudDetectionAdvanced provides training and inference
methods that are compatible with PyTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import pennylane as qml
import numpy as np
import torch


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


def _apply_layer(
    dev: qml.Device, params: FraudLayerParameters, *, clip: bool
) -> None:
    """Map a classical layer to a set of qubit rotations and entanglement."""
    theta = params.bs_theta
    phi = params.bs_phi
    # first layer of rotations
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=1)
    # phases as Z rotations
    qml.RZ(params.phases[0], wires=0)
    qml.RZ(params.phases[1], wires=1)
    # squeezing → RX + RZ
    qml.RX(_clip(params.squeeze_r[0], 5), wires=0)
    qml.RZ(_clip(params.squeeze_phi[0], 5), wires=0)
    qml.RX(_clip(params.squeeze_r[1], 5), wires=1)
    qml.RZ(_clip(params.squeeze_phi[1], 5), wires=1)
    # entanglement via CNOT
    qml.CNOT(wires=[0, 1])
    # displacement → rotation about Y
    qml.RY(_clip(params.displacement_r[0], 5), wires=0)
    qml.RY(_clip(params.displacement_r[1], 5), wires=1)
    # Kerr → ZZ (approximation)
    qml.CZ(wires=[0, 1])
    # final layer of rotations
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=1)
    qml.RZ(params.phases[0], wires=0)
    qml.RZ(params.phases[1], wires=1)
    qml.RX(_clip(params.squeeze_r[0], 5), wires=0)
    qml.RZ(_clip(params.squeeze_phi[0], 5), wires=0)
    qml.RX(_clip(params.squeeze_r[1], 5), wires=1)
    qml.RZ(_clip(params.squeeze_phi[1], 5), wires=1)
    qml.RY(_clip(params.displacement_r[0], 5), wires=0)
    qml.RY(_clip(params.displacement_r[1], 5), wires=1)
    qml.CZ(wires=[0, 1])


def build_variational_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: Optional[qml.Device] = None,
) -> qml.QNode:
    """Create a Pennylane QNode that evaluates the fraud‑detection circuit."""
    dev = device or qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor) -> torch.Tensor:
        # encode classical input as rotations on each wire
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        _apply_layer(dev, input_params, clip=False)
        for layer in layers:
            _apply_layer(dev, layer, clip=True)
        # measurement of the first wire
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionAdvanced:
    """Hybrid quantum‑classical fraud‑detection wrapper.

    The class accepts classical input features, runs them through a
    Pennylane variational circuit, and returns a fraud probability.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: Optional[qml.Device] = None,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> None:
        self.circuit = build_variational_program(
            input_params, layers, device=device
        )
        self.optimizer = torch.optim.Adam(
            self.circuit.parameters(), lr=lr
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.batch_size = batch_size

    def _train_one_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        logits = self.circuit(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                loss = self._train_one_batch(xb, yb)
                epoch_loss += loss * xb.size(0)
            epoch_loss /= len(loader.dataset)
            print(f"Epoch {epoch+1}/{self.epochs} loss: {epoch_loss:.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.circuit.eval()
        with torch.no_grad():
            logits = self.circuit(X)
            probs = torch.sigmoid(logits)
        return probs.cpu()
