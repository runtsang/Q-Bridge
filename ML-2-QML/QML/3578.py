from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Callable

class HybridFullyConnectedLayer:
    """
    Parameterised quantum circuit that mimics a fully‑connected layer.
    Supports batched evaluation and optional shot‑noise.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator_statevector")
        self.shots = shots
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.barrier()
        self._circuit.measure_all()
        self._parameters = [self.theta]

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute expectation of Z on the first qubit for a single parameter set.
        """
        circ = self._bind(thetas)
        state = Statevector.from_instruction(circ)
        pauli_z = SparsePauliOp.from_label("Z" + "I" * (self.n_qubits - 1))
        exp = state.expectation_value(pauli_z)
        return np.array([float(exp)])

class HybridEstimator:
    """
    Evaluates expectation values for a set of parameters and observables.
    """
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

def FCL(n_qubits: int = 1) -> HybridFullyConnectedLayer:
    """Return a hybrid fully connected quantum layer instance."""
    return HybridFullyConnectedLayer(n_qubits)

__all__ = ["HybridFullyConnectedLayer", "HybridEstimator", "FCL"]
