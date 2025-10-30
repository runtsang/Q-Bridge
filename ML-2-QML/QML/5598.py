"""Hybrid quantum classifier inspired by QCNN and EstimatorQNN."""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single convolution unit used in QCNN."""
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

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Pooling unit used in QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolution layer composed of conv_circuit blocks."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    param_index = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer composed of pool_circuit blocks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    param_index = 0
    for src, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index+3]), [src, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[EstimatorQNN, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a QCNN-inspired ansatz with a ZFeatureMap and return a EstimatorQNN.
    Parameters
    ----------
    num_qubits : int
        Number of qubits (input feature dimension).
    depth : int
        Number of convolution-pooling stages (unused but kept for compatibility).
    Returns
    -------
    qnn : EstimatorQNN
        Quantum neural network ready for training.
    encoding : Iterable[Parameter]
        List of input parameters.
    weight_sizes : Iterable[int]
        Number of trainable parameters in the ansatz.
    observables : List[SparsePauliOp]
        Observable used for measurement.
    """
    # Feature map
    feature_map = ZFeatureMap(num_qubits)
    # Ansatz: QCNN stack
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), inplace=True)
    # For deeper stages, repeat pattern (optional)
    for i in range(1, depth):
        sub_qubits = num_qubits // (2 ** i)
        ansatz.compose(conv_layer(sub_qubits, f"c{i+1}"), inplace=True)
        ansatz.compose(pool_layer(list(range(sub_qubits // 2)), list(range(sub_qubits // 2, sub_qubits)), f"p{i+1}"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observable: single Z on first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    # Estimator for forward pass
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    encoding = list(feature_map.parameters)
    weight_sizes = [len(ansatz.parameters)]
    observables = [observable]
    return qnn, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
