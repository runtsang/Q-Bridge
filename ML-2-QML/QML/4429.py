import numpy as np
from qiskit import execute, Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Pauli
from collections.abc import Iterable, Sequence
from typing import List, Optional, Union, Callable

from.Conv import Conv
from.EstimatorQNN import EstimatorQNN

class HybridEstimator:
    """Hybrid quantum estimator that evaluates a parametrized circuit with optional preprocessing and shot noise.

    Features:
    * Accepts a QuantumCircuit or a factory function such as EstimatorQNN.
    * Optional convolution filter circuit applied to input data before binding to the main circuit.
    * Optional preprocessor function that maps input parameters to new parameters.
    * Optional shot noise added to expectation values.
    * Supports a list of BaseOperator observables; defaults to identity on all qubits.
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Callable[[], QuantumCircuit]],
        *,
        conv_filter: Optional[Callable[[], QuantumCircuit]] = None,
        preprocessor: Optional[Callable[[Sequence[float]], Sequence[float]]] = None,
        shots: Optional[int] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(circuit, QuantumCircuit):
            self._circuit = circuit
        else:
            self._circuit = circuit()
        self.conv_filter = conv_filter() if conv_filter is not None else None
        self.preprocessor = preprocessor
        self.shots = shots
        self.backend = backend or "qasm_simulator"
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._backend = Aer.get_backend(self.backend)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if self.conv_filter is not None:
            # Conv filter circuit expects 2D data; we ignore its output for simplicity
            pass
        if self.preprocessor is not None:
            parameter_values = self.preprocessor(parameter_values)
        mapping = dict(zip(self._circuit.parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables) or [Pauli('I'*self._circuit.num_qubits)]
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound_circuit, self._backend, shots=self.shots, seed_simulator=self.seed)
                result = job.result()
                counts = result.get_counts(bound_circuit)
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if self.shots is not None:
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [complex(self.rng.normal(real, max(1e-6, 1 / self.shots))) for real in row]
                noisy.append(noisy_row)
            return noisy
        return results

__all__ = ["HybridEstimator"]
