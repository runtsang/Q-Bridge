import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class FastBaseEstimator:
    """Hybrid quantum estimator that evaluates either a plain circuit or a Qiskit EstimatorQNN.

    Parameters
    ----------
    circuit : Union[QuantumCircuit, QiskitEstimatorQNN]
        If a ``QuantumCircuit`` is provided the estimator evaluates expectation values
        of a list of ``BaseOperator`` observables.  If an ``EstimatorQNN`` instance is
        passed it is used directly and the observable is taken from the QNN.
    """
    def __init__(self, circuit: Union[QuantumCircuit, QiskitEstimatorQNN]) -> None:
        if isinstance(circuit, QiskitEstimatorQNN):
            self._circuit = circuit.circuit
            self._parameters = list(circuit.weight_params) + list(circuit.input_params)
            self._observable = circuit.observables[0]
        else:
            self._circuit = circuit
            self._parameters = list(circuit.parameters)
            self._observable = None

        self._estimator = Estimator()

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        if self._observable is not None:
            observables = [self._observable]
        else:
            observables = list(observables or [])
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[complex(rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots))
                  for val in row] for row in results]
        return noisy

# Factory helpers --------------------------------------------------------------

def QCNN() -> QiskitEstimatorQNN:
    """Return the Qiskit QCNN instance defined in the seed package."""
    from.QCNN import QCNN as _QCNN
    return _QCNN()

def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return the Qiskit EstimatorQNN instance defined in the seed package."""
    from.EstimatorQNN import EstimatorQNN as _EstimatorQNN
    return _EstimatorQNN()

__all__ = ["FastBaseEstimator", "QCNN", "EstimatorQNN"]
