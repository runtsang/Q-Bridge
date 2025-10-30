"""
Quantum sampler with fraud‑detection inspired variational layers.
Uses Pennylane to construct a hybrid circuit that can be executed on
a simulator or a real quantum device.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

# ------------------------------------------------------------------
# Fraud‑detection parameter container
# ------------------------------------------------------------------
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# ------------------------------------------------------------------
# Variational circuit construction
# ------------------------------------------------------------------
def _build_sampler_circuit(params: Sequence[float]) -> qml.QNode:
    """Simple two‑qubit sampler part using Ry and CNOT."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        qml.RY(inputs[0], 0)
        qml.RY(inputs[1], 1)
        qml.CNOT(0, 1)
        qml.RY(weights[0], 0)
        qml.RY(weights[1], 1)
        qml.CNOT(0, 1)
        qml.RY(weights[2], 0)
        qml.RY(weights[3], 1)
        return qml.probs(wires=[0, 1])

    return circuit

def _build_fraud_circuit(params: FraudLayerParameters, clip: bool = True) -> qml.QNode:
    """Photonic‑style layer translated into qubit operations."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Beam‑splitter equivalent: a rotation mixing the two wires
        qml.RX(params.bs_theta, 0)
        qml.RX(params.bs_phi, 1)
        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, i)
        # Squeezing → amplitude‑phase rotations
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.RY(_clip(r, 5) if clip else r, i)
            qml.RZ(_clip(phi, 5) if clip else phi, i)
        # Second beam‑splitter
        qml.RX(params.bs_theta, 0)
        qml.RX(params.bs_phi, 1)
        # Additional phases
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, i)
        # Displacements → rotations
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.RY(_clip(r, 5) if clip else r, i)
            qml.RZ(_clip(phi, 5) if clip else phi, i)
        # Kerr → controlled‑phase
        for i, k in enumerate(params.kerr):
            qml.CZ(i, (i+1)%2) if _clip(k, 1) > 0 else None
        return qml.probs(wires=[0, 1])

    return circuit

# ------------------------------------------------------------------
# Hybrid SamplerQNN wrapper
# ------------------------------------------------------------------
class SamplerQNN:
    """
    Quantum sampler that first runs a variational sampler circuit
    followed by a sequence of fraud‑detection style layers.
    """
    def __init__(
        self,
        sampler_weights: Sequence[float],
        fraud_layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.sampler_circuit = _build_sampler_circuit(sampler_weights)
        self.fraud_circuits = [
            _build_fraud_circuit(layer) for layer in fraud_layers
        ]

    def sample(self, inputs: Sequence[float]) -> np.ndarray:
        """
        Execute the hybrid circuit and return the final probability distribution.
        """
        probs = self.sampler_circuit(inputs, np.array([0.0, 0.0, 0.0, 0.0]))
        for circuit in self.fraud_circuits:
            probs = circuit(np.array(probs), np.array([0.0]))
        return probs

__all__ = ["FraudLayerParameters", "SamplerQNN"]
