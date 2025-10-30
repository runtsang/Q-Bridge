"""PennyLane implementation of a variational Gaussian‑like fraud‑detection circuit.

The quantum module builds a parameterised circuit that mimics the
photonic layer structure using rotations, displacements and Kerr
non‑linearities.  Expectation values of Pauli‑Z are returned and can
be directly combined with a classical post‑processing layer.

Key features
------------
* Uses PennyLane’s automatic differentiation to back‑propagate through
  the quantum circuit.
* Supports clipping of parameters to emulate photonic bounds.
* Provides a `FraudDetectionHybrid` class that exposes a `forward`
  method compatible with PyTorch tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np

# --------------------------------------------------------------------------- #
#  Parameter dataclass (identical to the classical counterpart)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# --------------------------------------------------------------------------- #
#  Quantum circuit builder
# --------------------------------------------------------------------------- #
def _apply_layer(qc: qml.QNode, params: FraudLayerParameters, clip: bool) -> None:
    """Add a single photonic‑style layer to the circuit."""
    # Beam splitter emulated by a rotation
    qml.Rot(params.bs_theta, params.bs_phi, 0.0, wires=0)
    qml.Rot(params.bs_theta, params.bs_phi, 0.0, wires=1)

    # Phase shifts
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    # Squeezing (approximated by rotation + CNOT)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Rot(_clip(r, 5.0) if clip else r,
                _clip(phi, 5.0) if clip else phi,
                0.0,
                wires=i)

    # Displacement (approximated by rotation)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Rot(_clip(r, 5.0) if clip else r,
                _clip(phi, 5.0) if clip else phi,
                0.0,
                wires=i)

    # Kerr non‑linearity (approximated by RZ)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0) if clip else k, wires=i)

    # Entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])

# --------------------------------------------------------------------------- #
#  QNode factory
# --------------------------------------------------------------------------- #
def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device | None = None,
) -> qml.QNode:
    """Return a PennyLane QNode that outputs a single expectation value."""
    if dev is None:
        dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray) -> np.ndarray:
        # Encode classical inputs into the circuit
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)

        # First layer (unclipped)
        _apply_layer(circuit, input_params, clip=False)

        # Remaining layers (clipped)
        for layer in layers:
            _apply_layer(circuit, layer, clip=True)

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit

# --------------------------------------------------------------------------- #
#  Hybrid class
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """Hybrid quantum‑classical fraud detector.

    The quantum part produces a raw score which is then passed through a
    lightweight classical sigmoid layer.  The class is fully differentiable
    when used with PyTorch tensors.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dev: qml.Device | None = None,
    ) -> None:
        self.qnode = build_fraud_detection_qnode(input_params, layers, dev)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the fraud probability for a batch of 2‑D inputs."""
        # Convert to PennyLane numpy for the QNode
        raw = self.qnode(inputs.numpy())
        # Classical sigmoid post‑processing
        return torch.sigmoid(torch.tensor(raw))

# --------------------------------------------------------------------------- #
#  Factory function
# --------------------------------------------------------------------------- #
def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device | None = None,
) -> FraudDetectionHybrid:
    """Convenience constructor mirroring the classical API."""
    return FraudDetectionHybrid(input_params, layers, dev)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_model"]
