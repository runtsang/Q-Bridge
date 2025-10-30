"""Quantum QCNN hybrid model implemented with Qiskit.

The model implements a quantum convolutional neural network with
convolution, pooling, and ansatz layers, and exposes a method
to evaluate the network on input data.  A lightweight estimator
class is provided for batch evaluation with optional shot noise.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from collections.abc import Iterable, Sequence
from typing import List, Iterable as IterableType
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: IterableType[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic estimator."""

    def evaluate(self, observables: IterableType[BaseOperator], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [complex(rng.normal(float(val.real), max(1e-6, 1 / shots)), 0) for val in row]
            noisy.append(noisy_row)
        return noisy


class QCNNHybrid:
    """Quantum convolutional neural network hybrid model."""

    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        """Construct the full QCNN circuit and wrap it in an EstimatorQNN."""
        # Feature map
        feature_map = ZFeatureMap(8, reps=1, insert_barriers=True, sparse=False)

        # Helper to build a 2‑qubit convolution circuit
        def conv_circuit(params):
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

        # Build a convolution layer over all qubits
        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            return qc

        # Helper to build a 2‑qubit pooling circuit
        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        # Build a pooling layer over specified source‑sink pairs
        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc.append(pool_circuit(params[param_index:param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            return qc

        # Assemble the ansatz
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Combine feature map and ansatz into a single circuit
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        # Observable for a single‑qubit measurement
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Wrap in EstimatorQNN for efficient evaluation
        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def evaluate(self, inputs: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
        """Return expectation values for given inputs and a single parameter set."""
        return self.qnn.evaluate(inputs, parameters)

    def predict(self, inputs: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
        """Return predictions for a single set of parameters."""
        return self.qnn.predict(inputs, parameters)


def QCNN() -> QCNNHybrid:
    """Factory returning the configured :class:`QCNNHybrid`."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "FastEstimator", "FastBaseEstimator", "QCNN"]
