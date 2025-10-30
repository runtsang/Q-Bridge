import pennylane as qml
import numpy as np
import torch
from dataclasses import dataclass
from typing import Iterable, Tuple

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

def build_fraud_detection_qnode(input_params: FraudLayerParameters,
                               layers: Iterable[FraudLayerParameters],
                               device: qml.Device = None):
    """Return a PennyLane QNode that applies the photonic layers sequentially."""
    if device is None:
        device = qml.device("default.qubit", wires=2)

    @qml.qnode(device, interface="torch")
    def qnode(x: torch.Tensor):
        # x is a 2‑element vector that we ignore; the circuit is deterministic
        for i, params in enumerate([input_params] + list(layers)):
            clip = i == 0
            # Beam splitter
            qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
            # Phase shifts
            for j, phase in enumerate(params.phases):
                qml.Rgate(phase, wires=j)
            # Squeezing
            for j, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                r_eff = _clip(r, 5.0) if clip else r
                qml.Sgate(r_eff, phi, wires=j)
            # Second beam splitter
            qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
            # Phase shifts again
            for j, phase in enumerate(params.phases):
                qml.Rgate(phase, wires=j)
            # Displacement
            for j, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                r_eff = _clip(r, 5.0) if clip else r
                qml.Dgate(r_eff, phi, wires=j)
            # Kerr
            for j, k in enumerate(params.kerr):
                k_eff = _clip(k, 1.0) if clip else k
                qml.Kgate(k_eff, wires=j)
        # Return expectation of photon number on mode 0 as a simple scalar output
        return qml.expval(qml.NumberOperator(0))
    return qnode

__all__ = ["FraudLayerParameters", "build_fraud_detection_qnode", "_clip"]
