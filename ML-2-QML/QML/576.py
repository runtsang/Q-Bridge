"""Quantum fully connected layer using a depth‑controlled variational circuit.

The circuit applies a sequence of Ry rotations and entangling CNOT gates
across ``n_qubits`` qubits.  Each rotation is parameterised by an entry
in ``thetas``.  The run method evaluates the circuit on a qasm simulator
and returns the mean Pauli‑Z expectation value over all qubits.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute


def FCL():
    class QuantumFullyConnectedLayer:
        def __init__(self, n_qubits: int = 1, depth: int = 2, shots: int = 1024):
            self.n_qubits = n_qubits
            self.depth = depth
            self.shots = shots
            self.backend = Aer.get_backend("qasm_simulator")
            self._build_circuit()

        def _build_circuit(self):
            self.circuit = QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits * self.depth)]
            idx = 0
            for _ in range(self.depth):
                for q in range(self.n_qubits):
                    self.circuit.ry(self.theta[idx], q)
                    idx += 1
                for q in range(self.n_qubits - 1):
                    self.circuit.cx(q, q + 1)
                self.circuit.barrier()
            self.circuit.measure_all()

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Execute the variational circuit with the supplied parameters.

            Parameters
            ----------
            thetas : Iterable[float]
                Iterable containing ``n_qubits * depth`` rotation angles.

            Returns
            -------
            np.ndarray
                Mean expectation value of Pauli‑Z over all qubits.
            """
            param_dict = {p: theta for p, theta in zip(self.theta, thetas)}
            job = execute(
                self.circuit,
                backend=self.backend,
                shots=self.shots,
                parameter_binds=[param_dict],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            total = 0.0
            for state, freq in counts.items():
                z_expect = sum(1 if bit == "0" else -1 for bit in state)
                total += z_expect * freq
            expectation = total / self.shots
            return np.array([expectation])

    return QuantumFullyConnectedLayer()


__all__ = ["FCL"]
