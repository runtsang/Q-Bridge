"""Quantum‑centric interface for the hybrid fully‑connected layer.

The quantum module mirrors the classical implementation but exposes
only the quantum‑specific components: the parameterized circuit,
encoding, and measurement.  It is designed to be imported
independently when only quantum functionality is required.

Typical usage:
    >>> from FCL__gen237 import FCL
    >>> model = FCL()
    >>> # Run quantum part explicitly
    >>> expectations = model.run_quantum(thetas, input_data)
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridFCLQuantum:
    """
    Quantum component of the hybrid fully‑connected layer.

    Parameters
    ----------
    n_features : int
        Number of qubits / input features.
    depth : int
        Depth of the variational ansatz.
    backend : qiskit.providers.Backend, optional
        Quantum backend; defaults to Aer qasm_simulator.
    shots : int, default=1024
        Number of shots for simulation.
    """

    def __init__(
        self,
        n_features: int = 1,
        depth: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.n_features = n_features
        self.depth = depth
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Parameter vectors
        self.encoding = ParameterVector("x", n_features)
        self.weights = ParameterVector("theta", n_features * depth)

        # Build circuit
        self.qc = QuantumCircuit(n_features)
        for param, qubit in zip(self.encoding, range(n_features)):
            self.qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(n_features):
                self.qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(n_features - 1):
                self.qc.cz(qubit, qubit + 1)

        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (n_features - i - 1))
            for i in range(n_features)
        ]

        self.qc.measure_all()

    def run(self, thetas: Iterable[float], input_data: Iterable[float]) -> np.ndarray:
        """Execute the circuit and return expectation values."""
        param_binds = [
            {self.encoding[i]: val for i, val in enumerate(input_data)},
            {self.weights[i]: val for i, val in enumerate(thetas)},
        ]
        job = qiskit.execute(
            self.qc,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.qc)

        expectation = np.zeros(len(self.observables))
        for state, cnt in counts.items():
            prob = cnt / self.shots
            bits = [int(b) for b in state[::-1]]
            for idx, _ in enumerate(self.observables):
                expectation[idx] += prob * (1 if bits[idx] == 0 else -1)
        return expectation


def FCL() -> HybridFCLQuantum:
    """Return the quantum part of the hybrid fully‑connected layer."""
    return HybridFCLQuantum()


__all__ = ["HybridFCLQuantum", "FCL"]
