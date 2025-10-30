"""
Hybrid fully‑connected layer implemented as a parameterized quantum circuit.

Implements a richer circuit:
  * Multiple qubits.
  * Entangling CNOT chain.
  * Parameterized Ry rotations.
  * Expectation value of the Pauli‑Z operator on the first qubit.

The run method accepts a list of thetas and returns the expectation
value as a NumPy array.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli


class FCLHybrid:
    """
    Quantum implementation of a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Backend, optional
        Quantum backend to execute on.  Defaults to the Aer QASM simulator.
    shots : int, default=1024
        Number of shots per execution.
    """

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """
        Create a parameterized circuit with entanglement.
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Parameterized Ry gates
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i, th in enumerate(self.theta):
            self.circuit.ry(th, i)
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit and compute the expectation of Pauli‑Z on qubit 0.

        Parameters
        ----------
        thetas : Iterable[float]
            List of rotation angles, one per qubit.

        Returns
        -------
        np.ndarray
            Array containing the expectation value.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} thetas, got {len(thetas)}")

        bound_circuit = self.circuit.bind_parameters(
            {th: val for th, val in zip(self.theta, thetas)}
        )
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Convert bitstrings to integer values and compute probabilities
        probs = np.array([counts.get(bs, 0) for bs in sorted(counts)]) / self.shots
        bitstrings = np.array([int(bs, 2) for bs in sorted(counts)])

        # Expectation of Z on qubit 0: +1 for |0>, -1 for |1>
        z_expectation = np.sum(probs * ((bitstrings & 1) * -2 + 1))
        return np.array([z_expectation])

    def train_step(self, thetas: Iterable[float], target: float = 0.0) -> float:
        """
        Dummy training helper that returns the squared error.

        Since we cannot perform back‑propagation on the simulator, this
        method simply evaluates the current error and can be used
        within a classical optimizer loop.
        """
        pred = self.run(thetas)[0]
        loss = (pred - target) ** 2
        return loss


__all__ = ["FCLHybrid"]
