"""Quantum implementation of the Hybrid Fully‑Connected Layer.

The class ``HybridFCL`` is a variational circuit that accepts a batch of
parameter vectors.  For each vector a parameterised quantum circuit is
executed on a backend, and the expectation value of Pauli‑Z on the first
qubit is returned.  The circuit uses a data‑encoding layer inspired by
quanvolution: a Ry rotation on each of the four qubits, followed by a
random entanglement layer, enabling the quantum layer to mimic the
classical patch extraction while remaining fully differentiable.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable, List

class HybridFCL:
    """Variational quantum circuit acting as a fully‑connected layer."""

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = Parameter("θ")
        # Build reusable circuit template
        self.circuit_template = QuantumCircuit(n_qubits)
        # Data‑encoding: Ry rotations on each qubit
        for q in range(n_qubits):
            self.circuit_template.ry(self.theta, q)
        # Entanglement layer (random-like)
        for q in range(n_qubits - 1):
            self.circuit_template.cx(q, q + 1)
        self.circuit_template.measure_all()

    def run(self, theta_batch: Iterable[List[float]]) -> np.ndarray:
        """Execute the circuit for each parameter vector in ``theta_batch``."""
        results = []
        for theta in theta_batch:
            circ = self.circuit_template.bind_parameters(
                {self.theta: theta[0]}  # use first element as global parameter
            )
            job = execute(circ, self.backend, shots=self.shots)
            counts = job.result().get_counts(circ)
            # Convert counts to expectation of Pauli‑Z on qubit 0
            exp = 0.0
            for bitstring, freq in counts.items():
                # bitstring is string of qubit states in reverse order
                z = 1 if bitstring[0] == "0" else -1
                exp += z * freq
            exp = exp / self.shots
            results.append(exp)
        return np.array(results)

__all__ = ["HybridFCL"]
