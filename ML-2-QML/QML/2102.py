"""Variational quantum circuit that emulates a fully‑connected layer.

The circuit contains a stack of rotate‑and‑entangle blocks whose parameters are
treated as the input ``thetas``.  The expectation value of a Pauli‑Z operator
on the first qubit is returned – this mirrors the classical network’s scalar
output while providing a genuine quantum contribution.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Pauli
from typing import Iterable, Sequence, List


class FullyConnectedLayer:
    """Parameterized quantum circuit mimicking a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.  The first qubit is used for the
        measurement; the remaining qubits provide entanglement.
    depth : int, default 1
        Number of repeat layers of RY rotations and CNOT entanglement.
    entanglement : str, {'full', 'linear'}, default 'full'
        Pattern of CNOT connections.  ``'full'`` applies a CNOT from each qubit
        to all following qubits; ``'linear'`` connects only neighbours.
    shots : int, default 1024
        Number of shots to run on a classical backend.  When a state‑vector
        backend is used this argument is ignored.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int = 1,
        entanglement: str = "full",
        shots: int = 1024,
        backend: Backend | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.shots = shots

        # Choose a backend that can return a state‑vector.
        self.backend = backend or AerSimulator()
        self.circuit = self._build_circuit()

    def _entangle(self, qc: QuantumCircuit, qreg: QuantumRegister) -> None:
        for i in range(self.n_qubits - 1):
            for j in range(i + 1, self.n_qubits):
                if self.entanglement == "full" or j == i + 1:
                    qc.cx(qreg[i], qreg[j])

    def _build_circuit(self) -> QuantumCircuit:
        qreg = QuantumRegister(self.n_qubits, "q")
        creg = ClassicalRegister(self.n_qubits, "c")  # unused for state‑vector
        qc = QuantumCircuit(qreg, creg)

        self.theta = ParameterVector("theta", self.n_qubits * self.depth)

        theta_idx = 0
        for _ in range(self.depth):
            qc.h(qreg)  # entangle with Hadamard
            qc.ry(self.theta[theta_idx : theta_idx + self.n_qubits], qreg)
            theta_idx += self.n_qubits
            self._entangle(qc, qreg)

        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[Sequence[float]]) -> np.ndarray:
        """Evaluate the circuit for a batch of parameter sets.

        Parameters
        ----------
        thetas
            Iterable of parameter vectors.  Each vector must have length
            ``n_qubits * depth``.
        Returns
        -------
        np.ndarray
            Mean Pauli‑Z expectation value on the first qubit over the batch.
        """
        expectations: List[float] = []

        for theta_vec in thetas:
            bound = {self.theta[i]: float(v) for i, v in enumerate(theta_vec)}
            bound_circuit = self.circuit.bind_parameters(bound)
            result = self.backend.run(bound_circuit, shots=self.shots).result()

            # If a state‑vector backend is used, compute expectation exactly.
            if isinstance(self.backend, AerSimulator):
                state = Statevector(result.get_statevector(bound_circuit))
                pauli_z = Pauli("Z" + "I" * (self.n_qubits - 1))
                exp = state.expectation_value(pauli_z)
                expectations.append(float(exp))
            else:
                # Use counts to estimate expectation.
                counts = result.get_counts(bound_circuit)
                probs = {k: v / self.shots for k, v in counts.items()}
                exp = 0.0
                for bitstring, p in probs.items():
                    z = 1 if bitstring[-1] == "0" else -1
                    exp += z * p
                expectations.append(exp)

        return np.array(expectations)

    @property
    def device(self):
        """Return the backend name for parity with the classical version."""
        return self.backend.name()


def FCL(
    n_qubits: int = 1,
    depth: int = 1,
    entanglement: str = "full",
    shots: int = 1024,
    backend: Backend | None = None,
) -> FullyConnectedLayer:
    """Convenience factory mimicking the original seed."""
    return FullyConnectedLayer(n_qubits, depth, entanglement, shots, backend)


__all__ = ["FullyConnectedLayer", "FCL"]
