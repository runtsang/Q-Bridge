"""Hybrid quantum estimator combining fast expectation evaluation and shot‑noise sampling.

The class extends the lightweight FastBaseEstimator from Qiskit and adds
optional shot‑noise simulation via a backend.  It also provides a
parameterised sampler circuit that mirrors the classical SamplerQNN.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler as QiskitSampler

ScalarObservable = BaseOperator


def _ensure_batch(values: Sequence[float]) -> list:
    """Return a list of parameter lists for the circuit."""
    return [list(values)]


class HybridEstimator:
    """
    Evaluate expectation values of a parametrised quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        A circuit containing all parameters that will be bound during
        evaluation.  The circuit can be a simple variational ansatz or a
        sampler circuit.

    Notes
    -----
    * The evaluate method supports multiple observables.
    * Optional shot noise can be added by specifying ``shots``; a
      QiskitSampler is used for sampling, while a Statevector is used for
      exact simulation.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    # ------------------------------------------------------------------
    # Core evaluation logic (inherited from FastBaseEstimator)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    # ------------------------------------------------------------------
    # Shot‑noise sampling (optional)
    # ------------------------------------------------------------------
    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Evaluate the circuit using a sampler to introduce measurement noise.

        Parameters
        ----------
        shots : int, optional
            Number of shots for the sampler.  If None, the exact statevector
            expectation values are returned.
        seed : int, optional
            Seed for the sampler backend.
        """
        if shots is None:
            return self.evaluate(observables, parameter_sets)

        sampler = QiskitSampler(seed=seed)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            exp = sampler.run(bound, observables=observables, shots=shots).result()
            # The sampler returns a dictionary of expectation values
            row = [exp.get_expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Classic sampler circuit (from SamplerQNN)
    # ------------------------------------------------------------------
    @staticmethod
    def SamplerQNN() -> QuantumCircuit:
        """
        Return a 2‑qubit parameterised circuit that mimics the classical
        sampler network.  The circuit consists of Ry rotations on each qubit
        followed by a CX and a second set of Ry rotations.
        """
        from qiskit.circuit import ParameterVector

        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return qc

__all__ = ["HybridEstimator"]
