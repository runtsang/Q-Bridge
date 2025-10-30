"""Quantum variational convolutional filter with trainable parameters.

The class uses a parameterised RX rotation circuit with entangling layers.
It can be trained via a classical optimiser (SPSA) to minimise a loss
expressed as the mean‑squared error between the circuit output and target labels.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RX


class QuantumConv:
    """
    Variational quantum convolutional filter.
    The `run` method accepts a 2‑D patch and returns the average probability
    of measuring |1> across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 200,
        threshold: float | None = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or AerSimulator()
        # Parameter vector for RX gates
        self.theta = ParameterVector("theta", self.n_qubits)
        # Build the circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        # Initial RX rotations
        self.circuit.compose(RX(self.theta), front=True, inplace=True)
        # Entangling layers
        for _ in range(2):
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
            self.circuit.barrier()
            self.circuit.compose(RX(self.theta), front=True, inplace=True)
        self.circuit.measure_all()
        # Current parameter bindings
        self.param_bindings: dict[qiskit.circuit.Parameter, float] | None = None

    def set_parameters(self, params: np.ndarray | list[float]) -> None:
        """Set the rotation angles of the circuit."""
        if len(params)!= self.n_qubits:
            raise ValueError("Parameter length mismatch")
        self.param_bindings = dict(zip(self.theta, params))

    def get_parameters(self) -> np.ndarray:
        """Return the current rotation angles."""
        if self.param_bindings is None:
            return np.zeros(self.n_qubits)
        return np.array([self.param_bindings[par] for par in self.theta])

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.flatten()
        if self.threshold is not None:
            flat = np.where(flat > self.threshold, np.pi, 0.0)
        else:
            flat = np.zeros_like(flat)
        # Bind parameters: use data‑dependent angles
        self.param_bindings = dict(zip(self.theta, flat))
        job = self.backend.run(self.circuit, shots=self.shots, parameter_binds=[self.param_bindings])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_counts = sum(counts.values())
        # Compute probability of measuring 1 on each qubit
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            for i in range(self.n_qubits):
                if bitstring[::-1][i] == "1":  # Qiskit stores bits in reverse order
                    probs[i] += cnt
        probs /= total_counts * self.n_qubits
        return probs.mean()

    def loss(self, data: np.ndarray, label: float) -> float:
        """Mean‑squared error loss for a single example."""
        pred = self.run(data)
        return (pred - label) ** 2

    def train(
        self,
        train_data: list[np.ndarray],
        train_labels: list[float],
        epochs: int = 10,
        lr: float = 0.01,
    ) -> None:
        """
        Train the circuit using SPSA optimiser.

        Parameters
        ----------
        train_data : list of np.ndarray
            List of 2‑D patches.
        train_labels : list of float
            Corresponding target values (e.g. 0 or 1).
        epochs : int
            Number of optimisation iterations.
        lr : float
            Learning rate (used internally by SPSA).
        """
        optimizer = SPSA(maxiter=epochs, eps=lr, alpha=0.602, gamma=0.101)
        initial_params = self.get_parameters()

        def objective(params):
            self.set_parameters(params)
            loss_sum = 0.0
            for d, l in zip(train_data, train_labels):
                loss_sum += self.loss(d, l)
            return loss_sum / len(train_data)

        new_params, _ = optimizer.optimize(initial_params, objective)
        self.set_parameters(new_params)


def Conv() -> QuantumConv:
    """Return a QuantumConv instance (compatible with the original API)."""
    return QuantumConv()
