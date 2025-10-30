"""Quantum implementation of the hybrid SamplerQNNGen335.

The circuit mirrors the QCNN structure: a feature map followed by
convolutional and pooling layers.  It can be used as a variational
sampler (via StatevectorSampler) or as a QNN for classification
(EstimatorQNN).  The class also exposes a lightweight FastBaseEstimator
for expectation‑value evaluation in state‑vector mode.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Sampler as StatevectorSampler, Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from collections.abc import Iterable, Sequence
from typing import List, Union, Callable

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Union[SparsePauliOp, str]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class SamplerQNNGen335:
    """
    Quantum sampler that implements a QCNN‑style architecture.
    The circuit consists of a ZFeatureMap followed by alternating
    convolution and pooling layers.  The class exposes both a
    variational sampler (via StatevectorSampler) and a QNN for
    classification (via EstimatorQNN).
    """
    def __init__(self, qubit_count: int = 8) -> None:
        self.qubit_count = qubit_count
        self.feature_map = ZFeatureMap(qubit_count)
        self.circuit = self._build_ansatz()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single convolution block on a pair of qubits."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Convolution layer operating on adjacent qubit pairs."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[idx:idx+3])
            qc.append(sub, [i, i+1])
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Pooling block on a pair of qubits."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int], prefix: str) -> QuantumCircuit:
        """Pooling layer mapping source qubits to sink qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[idx:idx+3])
            qc.append(sub, [src, sink])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full QCNN‑style ansatz."""
        qc = QuantumCircuit(self.qubit_count)
        # First convolution and pooling
        qc.compose(self._conv_layer(self.qubit_count, "c1"), inplace=True)
        qc.compose(self._pool_layer(list(range(self.qubit_count//2)),
                                    list(range(self.qubit_count//2, self.qubit_count)),
                                    "p1"), inplace=True)
        # Second layer on remaining qubits
        remaining = self.qubit_count // 2
        qc.compose(self._conv_layer(remaining, "c2"), inplace=True)
        qc.compose(self._pool_layer(list(range(remaining//2)),
                                    list(range(remaining//2, remaining)),
                                    "p2"), inplace=True)
        # Final convolution on last two qubits
        qc.compose(self._conv_layer(2, "c3"), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return qc

    def sampler(self) -> SamplerQNN:
        """Return a Qiskit Machine‑Learning SamplerQNN instance."""
        sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=self.circuit.decompose(),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            sampler=sampler,
        )

    def qnn(self) -> EstimatorQNN:
        """Return a Qiskit Machine‑Learning EstimatorQNN instance."""
        estimator = StatevectorEstimator()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.qubit_count - 1), 1)])
        return EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=estimator,
        )

    def evaluate(
        self,
        observables: Iterable[Union[SparsePauliOp, str]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Convenience wrapper around the FastBaseEstimator."""
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)


__all__ = ["SamplerQNNGen335", "FastBaseEstimator"]
