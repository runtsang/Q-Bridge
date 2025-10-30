"""Quantum sampler network using Pennylane."""
from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode
from pennylane import numpy as pnp
from pennylane.tape import QuantumTape
from typing import Callable, Iterable, Tuple


class SamplerQNN:
    """
    Variational quantum sampler with an entangled ansatz.
    The circuit outputs expectation values of Pauli‑Z on each qubit, which are
    interpreted as probabilities after a softmax.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        layers: int = 3,
        entanglement: str = "full",
        device: str | qml.Device = "default.qubit",
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        layers : int
            Number of variational layers.
        entanglement : str
            Entanglement pattern: 'full', 'circular', or 'none'.
        device : str | qml.Device
            Pennylane device or device name.
        seed : int | None
            Seed for device initialization.
        """
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.device = qml.device(device, wires=n_qubits, shots=1024, seed=seed)

        # Parameter shapes
        self.weight_shapes = {
            "weights": (layers, n_qubits, 3),  # RX, RY, RZ per qubit per layer
            "biases": (layers, n_qubits, 3),   # same as above
        }

        self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _entangle(self, i: int) -> None:
        """Apply entanglement pattern for layer i."""
        if self.entanglement == "full":
            for j in range(self.n_qubits):
                for k in range(j + 1, self.n_qubits):
                    qml.CNOT(wires=[j, k])
        elif self.entanglement == "circular":
            for j in range(self.n_qubits):
                qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
        # 'none' leaves qubits unentangled

    def _circuit(self, *params: Iterable[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Variational circuit returning expectation values of Pauli‑Z."""
        for layer, (weights, biases) in enumerate(params):
            for qubit in range(self.n_qubits):
                # Apply parameterized rotations
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
                # Add a small bias rotation
                qml.RZ(biases[layer, qubit, 0], wires=qubit)
            self._entangle(layer)

        # Measure expectation values of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Map classical inputs to a probability distribution via the quantum circuit.
        The inputs are encoded as Ry rotations before the variational layers.
        """
        # Encode inputs
        for qubit in range(self.n_qubits):
            qml.RY(inputs[qubit], wires=qubit)

        # Execute circuit
        raw = self.qnode(*self.weight_shapes["weights"], *self.weight_shapes["biases"])
        probs = np.exp(raw) / np.sum(np.exp(raw))
        return probs

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Convenience wrapper for sampling."""
        return self.sample(inputs)


def SamplerQNNFactory(**kwargs) -> SamplerQNN:
    """
    Factory that returns a SamplerQNN instance.
    The function name mirrors the legacy API to preserve compatibility.
    """
    return SamplerQNN(**kwargs)


__all__ = ["SamplerQNNFactory", "SamplerQNN"]
