"""Hybrid quantum estimator built on Qiskit.

The estimator evaluates a parametrised QuantumCircuit for a set of
observables.  It supports exact state‑vector evaluation as well as
shot‑based estimation via AerSimulator.  Observables are passed as
BaseOperator instances; PauliSumOp is treated specially to compute
expectation values from measurement counts.  This implementation
extends the original FastBaseEstimator, adding shot simulation and
compatibility with hybrid quantum‑classical subcircuits such as a
quanvolution filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp
from qiskit.providers.aer import AerSimulator


class HybridEstimator:
    """Evaluates a parametrised quantum circuit over multiple parameter sets
    and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised circuit that may contain any subcircuit, e.g. a
        quanvolution filter built with random two‑qubit layers.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._base_circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._base_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set and observable.

        If ``shots`` is None, a state‑vector simulator is used for exact
        expectation values.  Otherwise a Qiskit AerSimulator is employed
        and the expectation is estimated from measurement statistics.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circuit = self._bind(values)

            if shots is None:
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                sim = AerSimulator(method="qasm", shots=shots)
                job = sim.run(circuit)
                result = job.result()
                counts = result.get_counts(circuit)
                row = []
                for obs in observables:
                    if isinstance(obs, PauliSumOp):
                        exp = obs.convert_to_pauli_basis().eval(counts, shots)
                    else:
                        exp = obs.expectation_value(Statevector.from_counts(counts))
                    row.append(exp)
            results.append(row)

        return results


__all__ = ["HybridEstimator"]
