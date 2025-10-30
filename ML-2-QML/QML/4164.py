"""FraudDetectionModel – quantum implementation using PennyLane.

This module recreates the photonic fraud‑detection circuit as a variational
quantum circuit.  It also provides a quantum fully‑connected layer that
can be used as a drop‑in replacement for the classical FCL analogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F


# -------------------- 1. Layer definition ------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer (used to build the quantum circuit)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a value to [-bound, bound]."""
    return max(-bound, min(bound, value))


def _apply_layer(qc, params: FraudLayerParameters, clip: bool = False) -> None:
    """
    Translate photonic parameters into PennyLane gates.
    The mapping is intentionally simple: rotations + entanglement.
    """
    # Beam‑splitter analogue – two RY rotations
    qml.RY(params.bs_theta, wires=0, do_queue=False)
    qml.RY(params.bs_phi, wires=1, do_queue=False)

    # Phase shifts
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i, do_queue=False)

    # Squeezing – represented by RZ rotations (with optional clipping)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(r if not clip else _clip(r, 5.0), wires=i, do_queue=False)

    # Entanglement (BS + phase)
    qml.CNOT(wires=[0, 1], do_queue=False)

    # Displacement – additional RZ rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RZ(r if not clip else _clip(r, 5.0), wires=i, do_queue=False)

    # Kerr – simple RZ rotation (clipped)
    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1.0), wires=i, do_queue=False)


def build_fraud_detection_qnode(
    params_list: Sequence[FraudLayerParameters],
    shots: int = 1024,
) -> qml.QNode:
    """Return a PennyLane QNode that implements the fraud‑detection stack."""
    dev = qml.device("default.qubit", wires=2, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # The `inputs` tensor is interpreted as a set of parameters for the
        # final layer; it is ignored for the photonic part to keep the demo
        # lightweight.
        for i, params in enumerate(params_list):
            _apply_layer(circuit, params, clip=(i > 0))
        # Return expectation value of PauliZ on the first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit


# -------------------- 2. Quantum Fully‑Connected Layer -----------------------
class QuantumFullyConnectedLayer:
    """
    Variational circuit that emulates a fully‑connected quantum layer.
    Parameters are encoded as RY rotations on each qubit.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            for i, theta in enumerate(params):
                qml.RY(theta, wires=i, do_queue=False)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit and return a NumPy array."""
        params = torch.as_tensor(list(thetas), dtype=torch.float32)
        result = self.circuit(params)
        return result.detach().cpu().numpy()


# -------------------- 3. High‑level FraudDetectionQuantumModel ----------------
class FraudDetectionQuantumModel:
    """
    Quantum‑centric fraud‑detection model.
    It builds a variational circuit from photonic parameters and
    optionally appends a quantum fully‑connected layer.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_layers: Sequence[FraudLayerParameters],
        shots: int = 1024,
        use_fcl: bool = False,
    ) -> None:
        self.params_list = [input_params] + list(hidden_layers)
        self.circuit = build_fraud_detection_qnode(self.params_list, shots=shots)
        self.fcl = QuantumFullyConnectedLayer(shots=shots) if use_fcl else None

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Execute the fraud‑detection circuit.
        `inputs` are interpreted as parameters for the final quantum FCL
        if `use_fcl` is True; otherwise they are ignored.
        """
        out = self.circuit(inputs)
        if self.fcl is not None:
            out = torch.from_numpy(self.fcl.run(inputs.detach().cpu().numpy()))
        return out


__all__ = [
    "FraudLayerParameters",
    "_apply_layer",
    "build_fraud_detection_qnode",
    "QuantumFullyConnectedLayer",
    "FraudDetectionQuantumModel",
]
