"""Variational quantum classifier with backend flexibility and batch evaluation."""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers import Backend


class QuantumClassifierCircuit:
    """Wrapper that builds a variational ansatz and runs it on a specified backend."""

    def __init__(self, num_qubits: int, depth: int, backend: Backend | None = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("statevector_simulator")
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def evaluate(self, inputs: Iterable[List[float]]) -> np.ndarray:
        """Return expectation values of the Z observables for each input vector."""
        results = []
        for inp in inputs:
            param_binds = {param: val for param, val in zip(self.encoding, inp)}
            job = execute(self.circuit, self.backend, parameter_binds=param_binds, shots=1024)
            state = Statevector(job.result().get_statevector(self.circuit))
            exps = [state.expectation_value(obs).real for obs in self.observables]
            results.append(exps)
        return np.array(results)
