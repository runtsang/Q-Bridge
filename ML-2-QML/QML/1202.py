"""Quantum implementation of a fully‑connected layer.

The `FCL` class builds a depth‑controlled parameterized ansatz that emulates
a classical fully‑connected layer.  Each layer of the classical network
corresponds to a block of single‑qubit rotations followed by entangling CNOTs.
The `run` method accepts a flat list of angles and returns the expectation
value of Pauli‑Z on the last qubit, approximating the output of the
classical network.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator


class FCL:
    """Parameterized quantum circuit mimicking a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, equal to the number of hidden layers in the
        corresponding classical network.
    depth : int, optional
        Number of variational layers (defaults to 2).
    backend : str or qiskit.providers.BaseBackend, optional
        Backend to use for execution.  If ``None``, the Aer QASM simulator
        is used.
    shots : int, optional
        Number of measurement shots.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int = 2,
        backend: str | qiskit.providers.BaseBackend | None = None,
        shots: int = 1000,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = (
            Aer.get_backend("qasm_simulator")
            if backend is None
            else backend
        )
        self.theta = [Parameter(f"θ_{i}_{d}") for i in range(n_qubits) for d in range(depth)]
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the parameterized ansatz."""
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        self.circuit = QuantumCircuit(qr, cr)

        # Initial H gates
        self.circuit.h(qr)

        # Variational layers
        idx = 0
        for d in range(self.depth):
            # Single‑qubit rotations
            for q in range(self.n_qubits):
                self.circuit.ry(self.theta[idx], qr[q])
                idx += 1
            # Entangling CNOT chain
            for q in range(self.n_qubits - 1):
                self.circuit.cx(qr[q], qr[q + 1])

        # Measurement
        self.circuit.measure(qr, cr)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit and return the expectation of Z on the last qubit.

        Parameters
        ----------
        thetas
            Flat list of rotation angles matching the order of ``self.theta``.
        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value.
        """
        param_bind = {p: v for p, v in zip(self.theta, thetas)}
        bound_circ = self.circuit.bind_parameters(param_bind)

        job = execute(bound_circ, backend=self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circ)

        # Convert bitstrings to integers and compute expectation
        probs = np.array([counts.get(b, 0) for b in sorted(counts)]) / self.shots
        bitstrings = np.array([int(b, 2) for b in sorted(counts)])
        # Expectation of Z on last qubit: +1 for |0>, -1 for |1>
        z_values = 1 - 2 * (bitstrings & 1)
        expectation = np.sum(z_values * probs)

        return np.array([expectation])


__all__ = ["FCL"]
