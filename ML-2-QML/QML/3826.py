"""Qiskit implementation of a quantum evaluator for the hybrid fraud detection model.

The evaluator maps the 2‑D classical feature vector to a single rotation angle
for a one‑qubit circuit.  It returns the probability of measuring state |1|,
which serves as the quantum expectation value for the hybrid model.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter


class QuantumFraudCircuit:
    """Parameterised one‑qubit circuit used as the quantum layer in the hybrid model."""
    def __init__(self, backend: qiskit.providers.Backend | None = None, shots: int = 1024):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for each sample in ``inputs``.
        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, 2).  Only the first column is used as the rotation angle.
        Returns
        -------
        np.ndarray
            Shape (batch, 1) containing the probability of measuring |1|.
        """
        thetas = inputs[:, 0]
        results = []
        for theta in thetas:
            circ = self.circuit.bind_parameters({self.theta: theta})
            job = execute(circ, self.backend, shots=self.shots)
            result = job.result().get_counts(circ)
            counts = np.array(list(result.values()), dtype=float)
            probs = counts / self.shots
            # probability of measuring |1|
            prob_one = probs.sum()
            results.append(prob_one)
        return np.array(results).reshape(-1, 1)


__all__ = ["QuantumFraudCircuit"]
