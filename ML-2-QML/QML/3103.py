import numpy as np
from typing import Iterable, List, Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info.operators.base_operator import BaseOperator

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(_conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(_pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def QCNN() -> EstimatorQNN:
    """Return a variational QCNN implemented with EstimatorQNN."""
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    feature_map = ZFeatureMap(8)

    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

class HybridBaseEstimator:
    """Hybrid estimator that evaluates an EstimatorQNN with optional shot noise."""
    def __init__(self, qnn: EstimatorQNN) -> None:
        self.qnn = qnn

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        input_dim = len(self.qnn.input_params)
        results: List[List[complex]] = []
        inputs = [np.zeros(input_dim)] * len(parameter_sets)
        for params, inp in zip(parameter_sets, inputs):
            out = self.qnn.predict(inp, params)
            results.append(out)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy.append([rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row])
        return noisy

__all__ = ["HybridBaseEstimator", "QCNN"]
