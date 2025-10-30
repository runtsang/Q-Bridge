"""Hybrid estimator for Qiskit circuits, with optional autoencoder and EstimatorQNN support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised quantum circuit.

    Parameters
    ----------
    circuit:
        Either a `QuantumCircuit` or a `EstimatorQNN` instance.
    autoencoder:
        Optional circuit (e.g. a quantum autoencoder) that is composed before the main circuit.
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, QiskitEstimatorQNN],
        *,
        autoencoder: QuantumCircuit | None = None,
    ) -> None:
        self.circuit = circuit
        self.autoencoder = autoencoder
        if isinstance(circuit, QuantumCircuit):
            self._params = list(circuit.parameters)
        else:
            self._params = None

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("Parameter binding is only supported for bare QuantumCircuit.")
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _compose_autoencoder(self, circ: QuantumCircuit) -> QuantumCircuit:
        if self.autoencoder is None:
            return circ
        return circ.compose(self.autoencoder, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return a matrix of expectation values for each parameter set.

        If ``shots`` is supplied, a `Sampler` is used to estimate expectation values
        from measurement samples; otherwise a statevector is used for exact values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if isinstance(self.circuit, QuantumCircuit):
            if shots is None:
                for params in parameter_sets:
                    circ = self._compose_autoencoder(self._bind(params))
                    state = Statevector.from_instruction(circ)
                    row = [state.expectation_value(obs) for obs in observables]
                    results.append(row)
            else:
                sampler = QiskitSampler()
                for params in parameter_sets:
                    circ = self._compose_autoencoder(self._bind(params))
                    job = sampler.run(circ, shots=shots, seed=seed)
                    samples = job.result().samples
                    probs = {tuple(k): v / shots for k, v in samples.items()}
                    row = []
                    for obs in observables:
                        exp = 0.0
                        for bitstring, p in probs.items():
                            eig = 1.0
                            for idx, op in enumerate(obs.paulis):
                                if op[0] == "Z":
                                    eig *= 1 if bitstring[idx] == 0 else -1
                                # X and Y components average to 0 in computational basis
                            exp += eig * p
                        row.append(exp)
                    results.append(row)
        else:
            # EstimatorQNN path
            estimator = QiskitEstimator()
            for params in parameter_sets:
                circ = self.circuit
                if hasattr(circ, "assign_parameters"):
                    circ = circ.assign_parameters(dict(zip(circ.parameters, params)), inplace=False)
                if self.autoencoder is not None:
                    circ = circ.compose(self.autoencoder, inplace=False)
                job = estimator.run(circ, shots=shots, seed=seed)
                res = job.result()
                # EstimatorQNN returns a single observable value
                results.append([res.values[0]])
        return results


__all__ = ["FastBaseEstimator"]
