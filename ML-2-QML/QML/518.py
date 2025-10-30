"""Quantum hybrid fraud‑detection circuit implemented with PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pennylane as qml
import pennylane.numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters that describe a photonic layer and its variational hyper‑parameters."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    depth: int = 1
    entangle: bool = True


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudQuantumCircuit:
    """Variational photonic‑style circuit using PennyLane."""

    def __init__(
        self,
        dev: qml.Device,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.dev = dev
        self.input_params = input_params
        self.layers = list(layers)
        self.params = self._initialize_params()
        self.qnode = qml.QNode(self._circuit, dev, interface="torch")

    def _initialize_params(self) -> np.ndarray:
        """Create a flat array of trainable parameters for the ansatz."""
        param_list: List[np.ndarray] = []
        for layer in self.layers:
            depth = layer.depth
            # rotation angles for each mode
            param_list.append(np.random.uniform(-np.pi, np.pi, size=(depth, 2)))
            # displacement and squeezing parameters
            param_list.append(np.random.uniform(-np.pi, np.pi, size=(depth, 2)))
            if layer.entangle:
                param_list.append(np.random.uniform(-np.pi, np.pi, size=(depth, 2)))
        return np.concatenate([p.ravel() for p in param_list])

    def _circuit(self, inputs: np.ndarray, *params: np.ndarray) -> np.ndarray:
        """PennyLane circuit that emulates the photonic fraud‑detection model."""
        idx = 0
        # Encode classical input as displacement gates
        for i, val in enumerate(inputs):
            qml.Dgate(val, 0.0, wires=i)

        # Static input photonic layer
        self._apply_photonic_layer(self.input_params, clip=False)

        # Variational layers
        for layer in self.layers:
            depth = layer.depth
            for _ in range(depth):
                # Rotations
                for i in range(2):
                    qml.RX(params[idx], wires=i)
                    idx += 1
                # Displacements
                for i in range(2):
                    qml.Dgate(params[idx], 0.0, wires=i)
                    idx += 1
                # Entanglement
                if layer.entangle:
                    qml.CRX(params[idx], wires=[0, 1])
                    idx += 1
            # Fixed photonic operations after the ansatz
            self._apply_photonic_layer(layer, clip=True)

        # Measurement of a single qubit observable
        return qml.expval(qml.PauliZ(0))

    def _apply_photonic_layer(self, params: FraudLayerParameters, *, clip: bool) -> None:
        """Apply a photonic layer using PennyLane primitives."""
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r if not clip else _clip(r, 5), phi, wires=i)
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r if not clip else _clip(r, 5), phi, wires=i)
        for i, k in enumerate(params.kerr):
            qml.KerrGate(k if not clip else _clip(k, 1), wires=i)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the circuit on a single classical input vector."""
        np_inputs = np.array(inputs.detach().cpu().numpy())
        raw = self.qnode(np_inputs, *self.params)
        return torch.tensor(raw, dtype=torch.float32)

    def train(self, optimizer: qml.Optimizer, epochs: int = 10) -> None:
        """Simple training loop that updates the ansatz parameters."""
        for _ in range(epochs):
            loss = -self.forward(torch.tensor([0.0, 0.0]))  # placeholder
            optimizer.step(lambda: loss, self.params)

__all__ = ["FraudLayerParameters", "FraudQuantumCircuit"]
