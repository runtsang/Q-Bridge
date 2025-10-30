"""Quantum sampler network using a Pennylane variational circuit.

The circuit operates on three qubits and contains three layers of
parameterised single‑qubit rotations interleaved with linear‑chain CNOTs.
It returns the full probability distribution over all basis states,
facilitating sampling and hybrid training.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SamplerQNN:
    """
    A 3‑qubit variational sampler.

    Parameters
    ----------
    n_qubits : int, default=3
        Number of qubits in the circuit.
    wires : list[int] | None, default=None
        Wire indices.  Defaults to ``range(n_qubits)``.
    seed : int | None, default=None
        Random seed for weight initialization.
    """
    def __init__(self, n_qubits: int = 3, wires: list[int] | None = None,
                 seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

        if seed is not None:
            np.random.seed(seed)

        # Three rotation angles per qubit: [ry1, ry2, ry3]
        self.params = pnp.random.uniform(0, 2 * np.pi,
                                         (n_qubits, 3), requires_grad=True)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Encode classical inputs as rotation angles
            for i, wire in enumerate(self.wires):
                qml.RY(inputs[i], wires=wire)

            # First layer of rotations
            for i, wire in enumerate(self.wires):
                qml.RY(weights[i, 0], wires=wire)

            # Entangling layer 1
            for i in range(self.n_qubits - 1):
                qml.CNOT(self.wires[i], self.wires[i + 1])

            # Second layer of rotations
            for i, wire in enumerate(self.wires):
                qml.RY(weights[i, 1], wires=wire)

            # Entangling layer 2
            for i in range(self.n_qubits - 1):
                qml.CNOT(self.wires[i], self.wires[i + 1])

            # Third layer of rotations
            for i, wire in enumerate(self.wires):
                qml.RY(weights[i, 2], wires=wire)

            return qml.probs(wires=self.wires)

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the sampler network.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (n_qubits,) with angles in [0, 2π].

        Returns
        -------
        np.ndarray
            Probability distribution over all 2^n_qubits basis states.
        """
        return self.circuit(inputs, self.params)


def SamplerQNN() -> SamplerQNN:
    """
    Factory returning the upgraded quantum sampler instance.
    """
    return SamplerQNN()


__all__ = ["SamplerQNN"]
