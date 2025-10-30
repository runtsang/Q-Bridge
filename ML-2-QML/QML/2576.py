import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from collections.abc import Iterable, Sequence
from typing import List, Optional

class FastBaseEstimator:
    """Hybrid estimator that wraps a Qiskit circuit, optionally using a sampler and shot noise."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        use_sampler: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.use_sampler = use_sampler
        self.shots = shots
        self.seed = seed
        if use_sampler:
            self._sampler = StatevectorSampler()
        else:
            self._sampler = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
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

    def sample(self, parameter_set: Sequence[float], num_samples: int) -> List[int]:
        """Return measurement samples from the circuit, optionally using a statevector sampler."""
        if self.use_sampler:
            sampler = self._sampler
            bound_circ = self._bind(parameter_set)
            results = sampler.run(bound_circ, shots=num_samples).result()
            counts = results.get_counts()
            samples: List[int] = []
            for bitstring, freq in counts.items():
                samples.extend([int(bitstring[::-1], 2)] * freq)
            return samples
        else:
            bound_circ = self._bind(parameter_set)
            state = Statevector.from_instruction(bound_circ)
            probs = state.probabilities_dict()
            probs_list = [probs.get(f"{i:0{bound_circ.num_qubits}b}", 0.0) for i in range(2 ** bound_circ.num_qubits)]
            return list(np.random.choice(len(probs_list), size=num_samples, p=probs_list))

    @staticmethod
    def create_sampler_qnn(num_qubits: int = 2, depth: int = 2) -> QSamplerQNN:
        """Convenience factory for a simple parameterized SamplerQNN circuit."""
        inputs = ParameterVector("input", num_qubits)
        weights = ParameterVector("weight", num_qubits * depth)
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.ry(inputs[i], i)
        for d in range(depth):
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(num_qubits):
                qc.ry(weights[d * num_qubits + i], i)
        return QSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=StatevectorSampler())

__all__ = ["FastBaseEstimator"]
