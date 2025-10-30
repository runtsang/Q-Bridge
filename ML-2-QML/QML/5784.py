import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class SelfAttention:
    """
    Quantum variational circuit that mimics a self‑attention block.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.base_circuit = QuantumCircuit(n_qubits)

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits)
        # Rotate each qubit with a 3‑parameter Euler rotation
        for i in range(self.n_qubits):
            idx = 3 * i
            circ.rx(rotation_params[idx], i)
            circ.ry(rotation_params[idx + 1], i)
            circ.rz(rotation_params[idx + 2], i)
        # Entangle neighbouring qubits with controlled‑RX gates
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        return circ

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circ = self._build_circuit(rotation_params, entangle_params)
        circ.measure_all()
        job = qiskit.execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)


class FastBaseEstimator:
    """
    Evaluate expectation values of quantum observables for a circuit.
    """

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Wraps FastBaseEstimator to add shot‑noise sampling.
    """

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots))
                + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["SelfAttention", "FastBaseEstimator", "FastEstimator"]
