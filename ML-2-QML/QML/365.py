"""Quantum fraud detection model using PennyLane's photonic gates and variational training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pennylane as qml
from pennylane import numpy as np

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

class FraudDetectionModel:
    """Variational photonic fraud detection model."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: List[FraudLayerParameters],
        device: qml.Device | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = layers
        self.device = device or qml.device("strawberryfields.qsim", wires=2, shots=1000)
        self.params = self._flatten_params([input_params] + layers)
        self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _flatten_params(self, layers: List[FraudLayerParameters]) -> np.ndarray:
        """Flatten all layer parameters into a single 1‑D array."""
        flat = []
        for p in layers:
            flat.extend([p.bs_theta, p.bs_phi] +
                        list(p.phases) +
                        list(p.squeeze_r) + list(p.squeeze_phi) +
                        list(p.displacement_r) + list(p.displacement_phi) +
                        list(p.kerr))
        return np.array(flat, dtype=np.float64)

    def _circuit(self, params: np.ndarray) -> np.ndarray:
        """Apply the photonic circuit and return the photon‑number expectation of mode 0."""
        idx = 0
        # input layer
        self._apply_layer(params[idx:idx+14], clip=False)
        idx += 14
        # subsequent layers
        for _ in self.layers:
            self._apply_layer(params[idx:idx+14], clip=True)
            idx += 14
        # measurement: photon‑number expectation of mode 0
        return qml.expval(qml.PauliZ(0))  # placeholder for photon‑number

    def _apply_layer(self, params: np.ndarray, clip: bool) -> None:
        """Apply one photonic layer given a flat slice of parameters."""
        bs_theta, bs_phi = params[0:2]
        phase0, phase1 = params[2:4]
        sq_r0, sq_r1 = params[4:6]
        sq_phi0, sq_phi1 = params[6:8]
        disp_r0, disp_r1 = params[8:10]
        disp_phi0, disp_phi1 = params[10:12]
        k0, k1 = params[12:14]
        qml.BSgate(bs_theta, bs_phi, wires=[0, 1])
        qml.Rgate(phase0, wires=0)
        qml.Rgate(phase1, wires=1)
        qml.Sgate(sq_r0 if not clip else _clip(sq_r0, 5), sq_phi0, wires=0)
        qml.Sgate(sq_r1 if not clip else _clip(sq_r1, 5), sq_phi1, wires=1)
        qml.Dgate(disp_r0 if not clip else _clip(disp_r0, 5), disp_phi0, wires=0)
        qml.Dgate(disp_r1 if not clip else _clip(disp_r1, 5), disp_phi1, wires=1)
        qml.Kgate(k0 if not clip else _clip(k0, 1), wires=0)
        qml.Kgate(k1 if not clip else _clip(k1, 1), wires=1)

    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions for a batch of inputs."""
        # The circuit does not use the classical inputs directly; we would
        # normally encode them as displacements or other gates.  For the
        # purpose of this example we treat the circuit output as a raw
        # score and apply a sigmoid.
        scores = np.array([self.qnode(self.params) for _ in range(inputs.shape[0])])
        probs = 1 / (1 + np.exp(-scores))
        return (probs > threshold).astype(np.int64)

    def train(
        self,
        data_loader: Iterable,
        epochs: int,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Simple gradient‑descent training loop."""
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        for epoch in range(epochs):
            loss = 0.0
            for inputs, targets in data_loader:
                def loss_fn(p):
                    preds = self.predict(inputs, threshold=0.5)
                    return np.mean((preds - targets)**2)
                self.params, acc = opt.step_and_cost(loss_fn, self.params)
                loss += acc
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.4f}")

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
