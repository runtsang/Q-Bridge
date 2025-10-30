"""Hybrid estimator that can evaluate Qiskit quantum circuits.

The estimator can compute expectation values of arbitrary BaseOperator
observables and optionally return sampled probability distributions.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler, StatevectorSampler

class SamplerQNN:
    """A small parameterised quantum circuit that can be used as a sampler."""
    def __init__(self) -> None:
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

        self.circuit = qc
        self.input_params = inputs
        self.weight_params = weights

    def bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Bind a flat list of parameters to the circuit."""
        if len(param_values)!= len(self.input_params) + len(self.weight_params):
            raise ValueError("Parameter count mismatch for SamplerQNN.")
        mapping = dict(
            zip(list(self.input_params) + list(self.weight_params), param_values)
        )
        return self.circuit.assign_parameters(mapping, inplace=False)

class FastHybridEstimator:
    """Evaluate a parameterised quantum circuit for batches of parameters.

    Supports expectation value evaluation and optional sampling with a
    specified number of shots.  When ``shots`` is ``None`` the
    statevector is used for exact expectation values; otherwise a
    Qiskit ``Sampler`` is used to produce noisy estimates.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[Union[complex, float]]]:
        """
        Parameters
        ----------
        observables
            Iterable of BaseOperator observables. If None, a probability
            distribution over computational basis states is returned for
            each parameter set.
        parameter_sets
            Iterable of parameter vectors to evaluate.
        shots
            If provided, a noisy estimate of expectation values or a
            sampled probability distribution is returned.
        seed
            Seed for the Qiskit sampler to ensure reproducibility.
        """
        observables = list(observables) or []

        results: List[List[Union[complex, float]]] = []

        if shots is None:
            # Exact statevector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                if observables:
                    row = [state.expectation_value(obs) for obs in observables]
                else:
                    # Return full probability distribution
                    probs = state.probabilities_dict()
                    row = [probs.get(f"{i:02b}", 0.0) for i in range(4)]
                results.append(row)
        else:
            # Sampling with Qiskit Sampler
            sampler = Sampler()
            for values in parameter_sets:
                bound = self._bind(values)
                result = sampler.run(bound, shots=shots, seed=seed).result()
                counts = result.get_counts()
                if observables:
                    # Convert counts to expectation values
                    exp_vals = []
                    for obs in observables:
                        exp = 0.0
                        for bitstring, count in counts.items():
                            exp += obs.data[tuple(int(b) for b in bitstring)] * count
                        exp /= shots
                        exp_vals.append(exp)
                    results.append(exp_vals)
                else:
                    # Probability distribution
                    probs = {f"{i:02b}": cnt / shots for i, cnt in enumerate(counts.values())}
                    row = [probs.get(f"{i:02b}", 0.0) for i in range(4)]
                    results.append(row)

        return results

__all__ = ["FastHybridEstimator", "SamplerQNN"]
