"""Quantum fraud‑detection circuit built with PennyLane.

The implementation maps the photonic layer parameters onto a two‑qubit
device.  A parameter‑shift gradient is available, and the public API
mirrors the classical counterpart by exposing a ``FraudDetectionModel``
class that forwards to the underlying quantum circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import pennylane.numpy as np


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
    return max(-bound, min(bound, value))


class FraudDetectionQuantumCircuit:
    """Parameterised qubit circuit that emulates the photonic layers."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
        shots: int = 1024,
    ) -> None:
        self.device = device or qml.device("default.qubit", wires=2, shots=shots)
        self.input_params = input_params
        self.layer_params = list(layer_params)

        # The QNode accepts a flattened list of all parameters
        self.qnode = qml.QNode(self._circuit, self.device)

    def _apply_layer_params(self, params: tuple[float,...], clip: bool) -> None:
        """Apply a single layer given a flat tuple of 14 parameters."""
        # Unpack parameters
        bs_theta, bs_phi = params[0], params[1]
        phase0, phase1 = params[2], params[3]
        squeeze_r0, squeeze_r1 = params[4], params[5]
        squeeze_phi0, squeeze_phi1 = params[6], params[7]
        disp_r0, disp_r1 = params[8], params[9]
        disp_phi0, disp_phi1 = params[10], params[11]
        kerr0, kerr1 = params[12], params[13]

        # Mapping photonic gates to qubit operations.
        # Beam‑splitter → rotation on each qubit
        qml.RZ(bs_theta, wires=0)
        qml.RZ(bs_phi, wires=1)

        # Phase shifters
        qml.RZ(phase0, wires=0)
        qml.RZ(phase1, wires=1)

        # Squeezing → X‑rotation
        qml.RX(_clip(squeeze_r0, 5) if clip else squeeze_r0, wires=0)
        qml.RX(_clip(squeeze_r1, 5) if clip else squeeze_r1, wires=1)

        # Entanglement
        qml.CNOT(wires=[0, 1])

        # Displacement → Z‑rotation
        qml.RZ(_clip(disp_r0, 5) if clip else disp_r0, wires=0)
        qml.RZ(_clip(disp_r1, 5) if clip else disp_r1, wires=1)

        # Kerr non‑linearity → Z‑rotation
        qml.RZ(_clip(kerr0, 1) if clip else kerr0, wires=0)
        qml.RZ(_clip(kerr1, 1) if clip else kerr1, wires=1)

    def _circuit(self, *all_params: float) -> float:
        """Full circuit with all layers applied sequentially."""
        idx = 0
        # Input layer (unclipped)
        self._apply_layer_params(all_params[idx:idx + 14], clip=False)
        idx += 14
        # Hidden layers (clipped)
        for _ in self.layer_params:
            self._apply_layer_params(all_params[idx:idx + 14], clip=True)
            idx += 14
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def _flatten_all_params(self) -> list[float]:
        """Return a flat list of all parameters for the model."""
        all_params = []
        for p in [self.input_params] + self.layer_params:
            all_params.extend(
                [
                    p.bs_theta,
                    p.bs_phi,
                    p.phases[0],
                    p.phases[1],
                    p.squeeze_r[0],
                    p.squeeze_r[1],
                    p.squeeze_phi[0],
                    p.squeeze_phi[1],
                    p.displacement_r[0],
                    p.displacement_r[1],
                    p.displacement_phi[0],
                    p.displacement_phi[1],
                    p.kerr[0],
                    p.kerr[1],
                ]
            )
        return all_params

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Return the expectation value for each input sample."""
        # The quantum circuit does not depend on the classical input in this
        # simplified example, so the same expectation is returned for every sample.
        return np.array([self.qnode(*self._flatten_all_params()) for _ in inputs])

    def gradient(self, inputs: np.ndarray) -> np.ndarray:
        """Return the parameter‑shift gradient for each input sample."""
        grad_func = qml.grad(self.qnode)
        grads = []
        for _ in inputs:
            grads.append(grad_func(*self._flatten_all_params()))
        return np.array(grads)


class FraudDetectionModel:
    """Unified wrapper exposing a classical‑style API for the quantum circuit."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
        shots: int = 1024,
    ) -> None:
        self._circuit = FraudDetectionQuantumCircuit(
            input_params, layer_params, device=device, shots=shots
        )

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Delegate to the underlying quantum circuit."""
        return self._circuit.evaluate(inputs)

    def gradient(self, inputs: np.ndarray) -> np.ndarray:
        """Delegate to the underlying quantum circuit."""
        return self._circuit.gradient(inputs)


__all__ = ["FraudLayerParameters", "FraudDetectionQuantumCircuit", "FraudDetectionModel"]
