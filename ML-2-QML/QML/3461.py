"""Quantum estimator that supports expectation values, sampling, and a
SamplerQNN wrapper.  The implementation unifies the original
FastBaseEstimator with a quantum SamplerQNN and provides an optional
shot‑based sampler for empirical probability distributions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

class FastBaseEstimator:
    """Estimator for parametrised quantum circuits."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | None],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each observable.

        If ``shots`` is given, the circuit is sampled and a probability
        distribution over all computational‑basis states is returned
        for the ``None`` placeholder observable.
        """
        obs_list = list(observables) or [None]
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)

            if shots is not None:
                sampler = Sampler()
                result = sampler.run(bound, shots=shots).result()
                counts = result.get_counts()
                # build probability vector in lexicographic order
                probs = [counts.get(format(i, '0{}b'.format(bound.num_qubits)), 0) / shots
                         for i in range(2 ** bound.num_qubits)]
                row = [complex(p) for p in probs]
                results.append(row)
                continue

            state = Statevector.from_instruction(bound)
            row = [
                state.expectation_value(o) if o is not None else complex(state.data)
                for o in obs_list
            ]
            results.append(row)
        return results


def SamplerQNN() -> QSamplerQNN:
    """Return a qiskit‑machine‑learning SamplerQNN instance.

    The sampler implements a two‑qubit mixer with a single mixing layer
    followed by two weight rotations.  It is suitable for illustrating
    hybrid classical‑quantum training pipelines.
    """
    input_params = ParameterVector("input", 2)
    weight_params = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)
    qc.cx(0, 1)

    qc.ry(weight_params[0], 0)
    qc.ry(weight_params[1], 1)
    qc.cx(0, 1)

    qc.ry(weight_params[2], 0)
    qc.ry(weight_params[3], 1)

    sampler_qnn = QSamplerQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
    )
    return sampler_qnn


__all__ = ["FastBaseEstimator", "SamplerQNN"]
