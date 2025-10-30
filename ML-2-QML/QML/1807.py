import numpy as np
from typing import Iterable, List, Sequence, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.backend import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.transpiler import transpile
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.models.noise import NoiseModel

class FastBaseEstimator:
    """
    Evaluate expectation values for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit to be evaluated.
    backend : Backend, optional
        Execution backend. Defaults to AerSimulator.
    noise_model : NoiseModel, optional
        Noise model to apply during simulation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = backend or AerSimulator()
        self._noise_model = noise_model

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        If *shots* is None, uses state‑vector expectation; otherwise
        performs QASM sampling with the given shot count.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        else:
            bound_circuits = [self._bind(v) for v in parameter_sets]
            transpiled = transpile(bound_circuits, backend=self._backend, optimization_level=3)
            for circ in transpiled:
                if self._noise_model:
                    circ.noise_model = self._noise_model
                job = self._backend.run(circ, shots=shots)
                result = job.result()
                counts = result.get_counts()
                row = []
                for obs in observables:
                    exp = self._expectation_from_counts(counts, obs)
                    row.append(exp)
                results.append(row)
        return results

    def _expectation_from_counts(self, counts: dict, operator: BaseOperator) -> complex:
        """
        Naïve conversion of measurement counts to expectation value
        for operators diagonal in the computational basis.
        """
        if not isinstance(operator, Operator):
            raise ValueError("Unsupported operator type for sampling expectation.")
        mat = operator.data
        eigenvals = np.diag(mat)
        exp = 0.0
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            idx = int(bitstring, 2)
            exp += eigenvals[idx] * cnt / total
        return exp

__all__ = ["FastBaseEstimator"]
