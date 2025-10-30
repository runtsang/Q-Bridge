"""Quantum circuit combining a fully connected style rotation with a random entangling layer,
inspired by the FCL and Quanvolution examples."""

import qiskit
import numpy as np
from typing import Iterable


class HybridQuantumLayer:
    """Quantum circuit combining a fully connected rotation (Ry) with a random
    two‑qubit entangling layer, followed by measurement. The circuit is
    parameterized by a single rotation angle per run."""

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # Parameter for rotation
        self.theta = qiskit.circuit.Parameter("theta")
        # Fully connected style: H then Ry(theta)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        # Random entangling layer (mimicking RandomLayer from Quanvolution)
        for i in range(0, n_qubits - 1, 2):
            self.circuit.cx(i, i + 1)
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for each theta in thetas and return the
        expectation value of the measured computational basis states."""
        param_binds = [{self.theta: theta} for theta in thetas]
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])


__all__ = ["HybridQuantumLayer"]
