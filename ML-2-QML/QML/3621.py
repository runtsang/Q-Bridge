from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import Parameter
from typing import Iterable, List, Sequence, Iterable

class HybridFullyConnectedLayer:
    """
    Parameterised quantum circuit that behaves like a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the ansatz.
    backend : qiskit.providers.BaseBackend | None
        Execution backend; defaults to Aer's qasm_simulator.
    shots : int
        Number of shots per circuit execution.
    """

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = Parameter("theta")

        # Simple entangling ansatz: H on all qubits, then RX(theta) on each
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.rx(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for each theta in ``thetas`` and return
        the expectation value of the computational basis state number.
        """
        expectations: List[float] = []
        for theta in thetas:
            bound = self._circuit.assign_parameters({self.theta: theta})
            job = execute(bound, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            expectations.append(np.sum(states * probs))
        return np.array(expectations)

class HybridQuantumFastEstimator:
    """
    Quantum analogue of the FastEstimator that evaluates expectation values
    for given parameter sets and observables.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
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
        parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """
        Compute the expectation value of each observable for every parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["HybridFullyConnectedLayer", "HybridQuantumFastEstimator"]
