"""Quantum extension of the fast estimator, supporting variational circuits and multiple observables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parameterised circuit to evaluate.
    backend : Backend | None, optional
        The Qiskit backend used to simulate the circuit.  If ``None`` a
        :class:`qiskit.quantum_info.Statevector` simulation is used.
    noise_model : Optional[Any]
        A noise model to attach to the simulator backend.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        noise_model: Optional[object] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend
        self.noise_model = noise_model

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
        return_state: bool = False,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators whose expectation values are to be computed.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        return_state : bool, optional
            If ``True``, each row is extended with the statevector of the circuit after
            the parameters are bound.  This is useful for debugging but increases memory
            usage.

        Returns
        -------
        List[List[complex]]
            A list of rows; each row contains the expectation values for all
            observables (and, if requested, the statevector).
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            if self.backend is None:
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
                if return_state:
                    row.append(state.data)
            else:
                job = self.backend.run(bound_circuit, noise_model=self.noise_model)
                result = job.result()
                state = Statevector.from_dict(result.get_statevector(bound_circuit))
                row = [state.expectation_value(obs) for obs in observables]
                if return_state:
                    row.append(state.data)
            results.append(row)
        return results


__all__ = ["FastBaseEstimator"]
