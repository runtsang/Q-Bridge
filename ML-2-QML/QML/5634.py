"""Quantum implementation of the fraud‑detection circuit with parameter‑shift gradient."""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, List, Tuple

class FraudLayerParams:
    """Parameters for a single photonic layer (identical to the seed)."""
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

class FraudDetectionEnhanced:
    """Shared quantum class for fraud detection."""

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    @staticmethod
    def _apply_layer(params: FraudLayerParams, *, clip: bool) -> None:
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r if not clip else FraudDetectionEnhanced._clip(r, 5), phi, wires=i)
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r if not clip else FraudDetectionEnhanced._clip(r, 5), phi, wires=i)
        for i, k in enumerate(params.kerr):
            qml.Kgate(k if not clip else FraudDetectionEnhanced._clip(k, 1), wires=i)

    @classmethod
    def build_circuit(
        cls,
        input_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
        num_qubits: int = 2,
        device: str = "default.qubit",
        wires: List[int] | None = None,
    ) -> qml.QNode:
        """Construct a Pennylane QNode that evaluates the fraud‑detection circuit."""
        wires = wires or list(range(num_qubits))
        dev = qml.device(device, wires=wires)

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode classical inputs via displacement gates
            for i, val in enumerate(inputs):
                qml.Dgate(val, wires=i)
            # Fixed layers
            cls._apply_layer(input_params, clip=False)
            for layer in layers:
                cls._apply_layer(layer, clip=True)
            # Trainable displacement layer (four parameters)
            disp_r = [params[0], params[2]]
            disp_phi = [params[1], params[3]]
            for i in range(2):
                qml.Dgate(disp_r[i], disp_phi[i], wires=i)
            # Measurement
            return qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1))

        return circuit

    @staticmethod
    def parameter_shift_gradient(circuit, inputs: np.ndarray, params: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
        """Compute analytic gradient via parameter‑shift rule."""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            shifted_plus = params.copy()
            shifted_minus = params.copy()
            shifted_plus[i] += shift
            shifted_minus[i] -= shift
            f_plus = circuit(inputs, shifted_plus)
            f_minus = circuit(inputs, shifted_minus)
            grad[i] = (f_plus - f_minus) / (2 * np.sin(shift))
        return grad

__all__ = ["FraudLayerParams", "FraudDetectionEnhanced"]
