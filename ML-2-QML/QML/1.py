"""Variational quantum circuit with parameter‑shift gradient for a fully connected layer."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector


class FCL:
    """
    Variational circuit that emulates a fully‑connected layer.
    Supports multiple qubits, layers, and a parameter‑shift gradient.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used in the ansatz.
    n_layers : int
        Depth of the circuit.
    shots : int, optional
        Number of shots for measurement.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = ParameterVector("θ", length=n_qubits * n_layers)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a layered variational ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        theta_idx = 0
        for _ in range(self.n_layers):
            # Single‑qubit rotations
            for q in range(self.n_qubits):
                qc.ry(self.theta[theta_idx], q)
                theta_idx += 1
            # Entangling layer (nearest‑neighbour chain)
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc

    def _expectation(self, bound_counts: dict[str, int]) -> float:
        """Compute expectation of Z on the first qubit from measurement counts."""
        probs = {int(k, 2): v / self.shots for k, v in bound_counts.items()}
        expectation = sum((1 if (k >> (self.n_qubits - 1)) & 1 else -1) * p for k, p in probs.items())
        return expectation

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with given parameters and return the expectation
        value as a NumPy array.
        """
        param_bind = {self.theta[i]: v for i, v in enumerate(thetas)}
        bound_qc = self.circuit.bind_parameters(param_bind)
        transpiled = transpile(bound_qc, self.backend)
        qobj = assemble(transpiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp_val = self._expectation(counts)
        return np.array([exp_val])

    def parameter_shift_gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Estimate the gradient using the parameter‑shift rule.
        Returns a NumPy array of gradients matching the length of `thetas`.
        """
        shift = np.pi / 2
        grads = []
        base = self.run(thetas)[0]
        for i, theta in enumerate(thetas):
            theta_plus = list(thetas)
            theta_minus = list(thetas)
            theta_plus[i] += shift
            theta_minus[i] -= shift
            f_plus = self.run(theta_plus)[0]
            f_minus = self.run(theta_minus)[0]
            grad = (f_plus - f_minus) / 2
            grads.append(grad)
        return np.array(grads)

    def train_step(
        self,
        thetas: Iterable[float],
        target: float,
        lr: float = 1e-3,
    ) -> float:
        """
        Perform one gradient‑descent step using the parameter‑shift gradient.
        Returns the loss value.
        """
        pred = self.run(thetas)[0]
        loss = (pred - target) ** 2
        grads = self.parameter_shift_gradient(thetas)
        updated = [t - lr * g for t, g in zip(thetas, grads)]
        return loss, updated
