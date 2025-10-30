from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np
from pennylane import ops

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

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))

def _apply_layer(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    """Apply one photonic layer to the active wires."""
    # Beam splitter
    ops.BSgate(params.bs_theta, params.bs_phi) | (0, 1)

    # Phase shifters
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | i

    # Squeezing (clipped if requested)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        ops.Sgate(_clip(r, 5.0) if clip else r, phi) | i

    # Second beam splitter
    ops.BSgate(params.bs_theta, params.bs_phi) | (0, 1)

    # Phase shifters again
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | i

    # Displacement (clipped if requested)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        ops.Dgate(_clip(r, 5.0) if clip else r, phi) | i

    # Kerr non‑linearity
    for i, k in enumerate(params.kerr):
        ops.Kgate(_clip(k, 1.0) if clip else k) | i

def build_fraud_detection_circuit(
    param_sequence: Sequence[FraudLayerParameters],
) -> qml.QNode:
    """
    Construct a Pennylane QNode that encodes a 2‑dimensional input vector
    and returns the vector of Pauli‑Z expectation values after a sequence
    of photonic layers.

    The circuit is fully variational: the parameters in ``param_sequence``
    are held fixed during execution, but the circuit can be embedded
    in a hybrid training loop where the parameters are treated as
    trainable via Pennylane's autograd.
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray) -> np.ndarray:
        # Encode inputs as coherent displacements
        ops.Dgate(inputs[0]) | 0
        ops.Dgate(inputs[1]) | 1

        # Apply the sequence of layers
        for params in param_sequence:
            _apply_layer(params, clip=True)

        # Return expectation values of Pauli‑Z on both modes
        return [
            qml.expval(qml.PauliZ(i))
            for i in range(2)
        ]

    return circuit

__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit"]
