"""Quantum component of QCNNPlus – builds a QCNN ansatz and returns an EstimatorQNN."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_block(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in the QCNN ansatz."""
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


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Construct a convolutional layer that couples neighbouring qubits."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits - 1, 2):
        block = _conv_block(params[param_index : param_index + 3])
        qc.append(block, [i, i + 1])
        qc.barrier()
        param_index += 3
    # wrap‑around coupling
    if num_qubits % 2 == 0:
        block = _conv_block(params[param_index : param_index + 3])
        qc.append(block, [num_qubits - 2, 0])
    return qc


def _pool_block(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block that reduces correlations."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pool two qubits into one, discarding the sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        block = _pool_block(params[param_index : param_index + 3])
        qc.append(block, [src, snk])
        qc.barrier()
        param_index += 3
    return qc


def build_qcnn_ansatz(num_qubits: int, depth: int = 3) -> QuantumCircuit:
    """Build a QCNN ansatz with alternating convolution and pooling layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the feature map.
    depth : int, default 3
        Number of conv‑pool pairs.
    """
    qc = QuantumCircuit(num_qubits)
    # Initial convolution
    qc.compose(conv_layer(num_qubits, "c0"), inplace=True)
    # Pooling to halve the qubit count
    pool_qc = pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p0")
    qc.compose(pool_qc, inplace=True)

    current_qubits = num_qubits // 2
    for d in range(1, depth):
        qc.compose(conv_layer(current_qubits, f"c{d}"), inplace=True)
        pool_qc = pool_layer(
            list(range(current_qubits // 2)),
            list(range(current_qubits // 2, current_qubits)),
            f"p{d}",
        )
        qc.compose(pool_qc, inplace=True)
        current_qubits //= 2
    return qc


def get_qcnn_qnn(
    num_qubits: int,
    feature_map: ZFeatureMap | None = None,
    conv_depth: int = 3,
    pool_depth: int = 3,
    estimator: Estimator | None = None,
) -> EstimatorQNN:
    """Create an EstimatorQNN that implements the QCNN architecture.

    The returned QNN can be passed to the :class:`~QCNNPlus` class in the
    classical module.  The feature map is a ZFeatureMap by default.
    """
    if estimator is None:
        estimator = Estimator()

    if feature_map is None:
        feature_map = ZFeatureMap(num_qubits)

    ansatz = build_qcnn_ansatz(num_qubits, depth=conv_depth)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = [
    "conv_layer",
    "pool_layer",
    "build_qcnn_ansatz",
    "get_qcnn_qnn",
]
