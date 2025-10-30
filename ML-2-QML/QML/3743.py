"""Hybrid fully‑connected layer with a parameterised quantum circuit.

The quantum implementation follows the spirit of the original FCL seed
but enriches the circuit with a layer of parallel Ry rotations and
full‑to‑full entanglement to expose richer feature maps.  The
``run`` method accepts a list of real parameters and returns the
expectation value of the measured Pauli‑Z string.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

__all__ = ["HybridFCL"]

class HybridFCL:
    """
    Quantum implementation of a fully‑connected layer.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("θ")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """
        Constructs a circuit with:
          * Hadamard pre‑rotation
          * Parallel Ry(θ) rotations
          * Full‑to‑full CNOT entanglement (dense connectivity)
          * Measurement of all qubits
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(self.n_qubits))
        # Dense entanglement
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                self.circuit.cx(i, j)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Execute the circuit for each θ in ``thetas`` and return the
        expectation value of the Pauli‑Z string.

        Parameters
        ----------
        thetas : array‑like of float
            Parameters to bind to the circuit.

        Returns
        -------
        np.ndarray
            1‑D array of expectation values, one per input θ.
        """
        if isinstance(thetas, list):
            thetas = np.array(thetas, dtype=np.float64)
        expectations = []
        for theta in thetas:
            bound_circuit = self.circuit.bind_parameters({self.theta: theta})
            job = execute(bound_circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circuit)
            probs = np.array(list(counts.values())) / self.shots
            # Encode measurement outcomes as ±1 (|0⟩→+1, |1⟩→−1) for each qubit
            outcomes = np.array([int(k, 2) for k in counts.keys()])
            signs = (-1) ** outcomes
            expectation = np.sum(signs * probs)
            expectations.append(expectation)
        return np.array(expectations, dtype=np.float64)
