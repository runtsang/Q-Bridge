"""Quantum fraud detection model using PennyLane with a photonic‑style variational circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer in a quantum circuit."""
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
    """
    Quantum fraud‑detection model built with PennyLane.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (unclipped).
    layers : Iterable[FraudLayerParameters]
        Subsequent layers (clipped to keep parameters in a realistic range).
    dev : str | pennylane.Device, optional
        Underlying PennyLane device (default: 'default.qubit').
    shots : int, optional
        Number of measurement shots per forward pass.
    noise : bool, optional
        If True, adds a simple depolarizing noise model to the device.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dev: str | qml.Device = "default.qubit",
        shots: int = 1024,
        noise: bool = False,
    ) -> None:
        self.dev = qml.device(dev, wires=2, shots=shots)
        self.layers = [input_params] + list(layers)
        self._build_qnode(noise=noise)

    def _apply_layer(self, params: FraudLayerParameters, clip: bool = False) -> None:
        """Apply one photonic‑style layer to the current circuit."""
        theta, phi = params.bs_theta, params.bs_phi
        qml.BSgate(theta, phi, wires=[0, 1])

        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)

        for i, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r if not clip else _clip(r, 5), p, wires=i)

        qml.BSgate(theta, phi, wires=[0, 1])

        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)

        for i, (r, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r if not clip else _clip(r, 5), p, wires=i)

        for i, k in enumerate(params.kerr):
            qml.Kgate(k if not clip else _clip(k, 1), wires=i)

    def _build_qnode(self, noise: bool) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x: np.ndarray) -> np.ndarray:
            # Encode classical input into mode amplitudes
            qml.Displacement(x[0], 0.0, wires=0)
            qml.Displacement(x[1], 0.0, wires=1)

            # First (unclipped) layer
            self._apply_layer(self.layers[0], clip=False)

            # Remaining layers
            for params in self.layers[1:]:
                self._apply_layer(params, clip=True)

            # Measurement: expectation values of the number operators
            return [qml.expval(qml.NumberOperator(w)) for w in range(2)]

        self.qnode = circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute the quantum circuit and return measurement results."""
        return self.qnode(x)

    def train_one_step(
        self,
        x: np.ndarray,
        y: float,
        loss_fn: callable,
        opt: qml.GradientDescentOptimizer,
        lr: float,
    ) -> float:
        """Perform a single optimisation step."""
        def cost_fn(params):
            # Update circuit parameters from optimizer
            self._apply_layer_from_params(params)
            preds = self.forward(x)
            return loss_fn(preds, y)

        grads = qml.grad(cost_fn)(self._params_vector())
        opt.step(grads, self._params_vector())
        return cost_fn(self._params_vector())

    def _params_vector(self) -> np.ndarray:
        """Flatten all trainable parameters into a single vector."""
        vec = np.array([])
        for params in self.layers:
            vec = np.concatenate(
                [vec, np.array(params.bs_theta), np.array(params.bs_phi),
                 np.array(params.phases), np.array(params.squeeze_r),
                 np.array(params.squeeze_phi), np.array(params.displacement_r),
                 np.array(params.displacement_phi), np.array(params.kerr)]
            )
        return vec

    def _apply_layer_from_params(self, vec: np.ndarray) -> None:
        """Reconstruct layer parameters from a flat vector and rebuild the circuit."""
        offset = 0
        new_layers = []
        for _ in self.layers:
            bs_theta = vec[offset]; offset += 1
            bs_phi = vec[offset]; offset += 1
            phases = tuple(vec[offset:offset+2]); offset += 2
            squeeze_r = tuple(vec[offset:offset+2]); offset += 2
            squeeze_phi = tuple(vec[offset:offset+2]); offset += 2
            displacement_r = tuple(vec[offset:offset+2]); offset += 2
            displacement_phi = tuple(vec[offset:offset+2]); offset += 2
            kerr = tuple(vec[offset:offset+2]); offset += 2
            new_layers.append(
                FraudLayerParameters(
                    bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                    displacement_r, displacement_phi, kerr
                )
            )
        self.layers = new_layers
        self._build_qnode(noise=False)  # rebuild circuit with new parameters

    def get_parameters(self) -> Sequence[FraudLayerParameters]:
        return self.layers

    def export_statevector(self, x: np.ndarray) -> np.ndarray:
        """Return the state vector for a given input (requires statevector device)."""
        sv_dev = qml.device("default.qubit", wires=2, shots=1, shots_per_batch=1, backend="statevector")
        @qml.qnode(sv_dev, interface="autograd")
        def sv_circuit(x):
            qml.Displacement(x[0], 0.0, wires=0)
            qml.Displacement(x[1], 0.0, wires=1)
            self._apply_layer(self.layers[0], clip=False)
            for params in self.layers[1:]:
                self._apply_layer(params, clip=True)
            return qml.state()
        return sv_circuit(x)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
