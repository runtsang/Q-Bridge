"""Hybrid estimator for quantum circuits with optional classical filter support.
It evaluates expectation values of observables for parameter sets, allowing
shot‑count control and a fallback classical filter when no quantum circuit
is provided."""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Dict, Any

from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


# --------------------------------------------------------------------------- #
# Classical filter implementation (fallback)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum‑inspired filter that runs a parameterised circuit on small image patches."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 0.0):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [QuantumCircuit().Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += QuantumCircuit().random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Run the circuit on a 2D data patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


def Conv() -> QuanvCircuit:
    """Return a default QuanvCircuit configured with a 2×2 filter."""
    return QuanvCircuit(kernel_size=2, shots=100, threshold=127)


# --------------------------------------------------------------------------- #
# Base quantum estimator utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters) if circuit is not None else []

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Hybrid quantum estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimatorGen137(FastEstimator):
    """
    Hybrid estimator that evaluates either a quantum circuit or a classical
    filter.  When a circuit is provided, expectation values of the supplied
    observables are returned.  If no circuit is given, the estimator falls
    back to running a classical filter (QuanvCircuit) on the data.
    """
    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        *,
        shots: int = 100,
        threshold: float = 0.0,
        backend=None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        if circuit is not None:
            self._parameters = list(circuit.parameters)
        else:
            self._parameters = []

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if self.circuit is None:
            raise RuntimeError("No quantum circuit supplied for binding.")
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | Callable[[Sequence[float]], float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            if self.circuit is not None:
                state = Statevector.from_instruction(self._bind(params))
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Assume observables are callables that accept the parameter set
                row = [obs(params) for obs in observables]
            results.append(row)
        return results


__all__ = ["FastBaseEstimatorGen137", "QuanvCircuit", "Conv"]
