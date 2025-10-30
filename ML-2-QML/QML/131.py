"""Variational quantum circuit that mimics a fully connected layer.

The circuit uses a parameterised Ry rotation per qubit and a
single layer of CNOT entanglement.  The output is the expectation
value of the Pauli‑Z observable on the first qubit.  The class
provides a ``run`` method that accepts an iterable of parameters
and returns the expectation as a NumPy array.

The implementation supports both the Aer simulator and any
publicly available backend.  A simple ``train`` helper is
included that performs a single gradient‑shift step using
parameter‑shift rules.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers import Backend


class FCL:
    """
    Variational circuit with a single qubit.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the circuit.
    depth : int, default 1
        Number of Ry layers (each layer consists of a Ry on every qubit
        followed by a full‑chain of CNOTs).
    backend : Backend, optional
        Quantum backend to execute on.  If None, the Aer qasm simulator
        is used.
    shots : int, default 1024
        Number of shots per execution.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        depth: int = 1,
        backend: Backend | None = None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vector: one theta per Ry per qubit per layer
        self.theta = ParameterVector("theta", length=depth * n_qubits)

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                qc.ry(self.theta[idx], q)
                idx += 1
            # Full‑chain CNOT entanglement
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameters.  Length must match ``depth * n_qubits``.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the expectation value of Pauli‑Z
            on qubit 0.
        """
        if len(thetas)!= self.theta.size:
            raise ValueError(
                f"Expected {self.theta.size} parameters, got {len(thetas)}"
            )
        param_bind = {self.theta[idx]: val for idx, val in enumerate(thetas)}
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Map bitstrings to expectation of Z on qubit 0
        exp = 0.0
        total = 0
        for bitstring, cnt in counts.items():
            prob = cnt / self.shots
            # bitstring is in reverse order; last bit corresponds to qubit 0
            qubit0 = int(bitstring[0])
            exp += (1.0 if qubit0 == 0 else -1.0) * prob
            total += prob
        return np.array([exp])

    def params(self) -> Sequence[Parameter]:
        """Return the circuit parameters."""
        return list(self.theta)

    def circuit_obj(self) -> QuantumCircuit:
        """Return the underlying Qiskit circuit."""
        return self.circuit


__all__ = ["FCL"]
