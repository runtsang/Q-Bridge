"""ConvHybridQuantum: Variational circuit for a 2‑D filter."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ConvHybridQuantum:
    """
    Quantum implementation of a 2‑D filter using a parameter‑tuned
    variational circuit.  The circuit encodes the input pixel values
    as rotation angles and outputs the mean probability of measuring
    |1⟩ across all qubits.  It is designed to be fused with the
    classical ConvHybrid via a learnable gate.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
        lr: float = 0.01,
    ) -> None:
        """
        Args:
            kernel_size: Size of the square filter.
            shots: Number of measurement shots for statevector sampling.
            threshold: Pixel value threshold to decide rotation angle.
            lr: Learning rate for the variational parameters.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        # Device: GPU if available, otherwise CPU
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)

        # Initialize variational parameters (one per qubit)
        self.params = pnp.random.uniform(0, 2 * np.pi, self.n_qubits, requires_grad=True)

        # Optimizer for the variational parameters
        self.opt = qml.AdamOptimizer(stepsize=lr)

    def _encode(self, data: np.ndarray) -> None:
        """
        Encode the 2‑D data into rotation angles.
        Pixels > threshold are encoded as π, otherwise 0.
        """
        flat = data.flatten()
        self.theta = np.where(flat > self.threshold, np.pi, 0.0)

    def _variational_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Define a simple variational circuit: Ry rotations per qubit,
        followed by a chain of CNOT gates, and a final layer of Ry
        rotations.  Returns the statevector.
        """
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Encode data
            for i, angle in enumerate(self.theta):
                qml.Ry(angle, wires=i)
            # Variational layers
            for i in range(self.n_qubits):
                qml.Ry(params[i], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Final variational layer
            for i in range(self.n_qubits):
                qml.Ry(params[i], wires=i)
            return qml.probs(wires=range(self.n_qubits))

        return circuit()

    def step(self, data: np.ndarray, target: float) -> None:
        """
        Perform one optimization step given a target output.
        The loss is mean‑squared error between the circuit output
        and the target.

        Args:
            data: 2‑D input array of shape (kernel_size, kernel_size).
            target: Desired scalar output.
        """
        self._encode(data)

        def loss_fn(params):
            probs = self._variational_circuit(params)
            mean_prob = probs.mean()
            return (mean_prob - target) ** 2

        self.params = self.opt.step(loss_fn, self.params)

    def run(self, data: np.ndarray) -> float:
        """
        Compute the output of the variational circuit for the given data.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            float: Mean probability of measuring |1⟩ across qubits.
        """
        self._encode(data)
        probs = self._variational_circuit(self.params)
        return float(probs.mean())

    def train(
        self,
        dataset: list[np.ndarray],
        targets: list[float],
        epochs: int = 10,
    ) -> None:
        """
        Train the variational parameters over a dataset.

        Args:
            dataset: List of 2‑D arrays.
            targets: Corresponding target outputs.
            epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            for data, target in zip(dataset, targets):
                self.step(data, target)
            print(f"Epoch {epoch + 1} complete.")

__all__ = ["ConvHybridQuantum"]
