"""Hybrid fraud‑detection model – quantum branch using Pennylane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters for a variational quantum layer."""
    n_qubits: int
    n_rotations: int  # number of single‑qubit rotations per qubit
    entanglement: str  # 'cnot' or 'cx' or 'cz'
    use_rotation: bool = True
    use_entanglement: bool = True
    clip: float | None = None  # clip gate angles to this bound


def _clip_params(params: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(params, -bound, bound)


def _variational_layer(params: FraudLayerParameters, state: Sequence[float]) -> None:
    """Apply a single variational layer to the quantum state."""
    if params.use_rotation:
        for i in range(params.n_qubits):
            for _ in range(params.n_rotations):
                theta = state.pop(0)
                phi = state.pop(0)
                if params.clip is not None:
                    theta = _clip_params(theta, params.clip)
                    phi = _clip_params(phi, params.clip)
                qml.Rot(theta, phi, 0, wires=i)

    if params.use_entanglement:
        if params.entanglement == "cnot":
            for i in range(params.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif params.entanglement == "cx":
            for i in range(params.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif params.entanglement == "cz":
            for i in range(params.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])


def build_fraud_detection_qnode(
    dev: qml.Device,
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a Pennylane QNode implementing the fraud‑detection circuit."""

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # Encode classical inputs as rotations
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)

        # Flatten parameter list for all layers
        all_params = []
        for layer in layers:
            # Each layer adds 2 * n_qubits * n_rotations parameters
            n = layer.n_qubits * layer.n_rotations * 2
            layer_params = torch.randn(n, requires_grad=True)
            all_params.extend(layer_params)

        # Apply layers sequentially
        idx = 0
        for layer in layers:
            # Slice parameters for this layer
            n_params = layer.n_qubits * layer.n_rotations * 2
            layer_state = all_params[idx : idx + n_params]
            idx += n_params
            _variational_layer(layer, layer_state)

        return qml.expval(qml.PauliZ(0))

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_qnode"]
