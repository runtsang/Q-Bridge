"""Quantum implementation of a QCNN with an embedded classifier ansatz.

The circuit emulates the convolution‑pooling structure of the original QCNN
and appends a variational classifier on top of the reduced qubit set.
The resulting EstimatorQNN can be used as a quantum neural network
in place of the classical model.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Iterable, Tuple

# ----------------------------------------------------------------------
# Variational classifier ansatz
# ----------------------------------------------------------------------
def _build_classifier_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Return a variational classifier circuit with Ry and CZ layers."""
    encoding = ParameterVector("x", num_qubits)
    weights   = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    return circuit

# ----------------------------------------------------------------------
# Convolution and pooling primitives
# ----------------------------------------------------------------------
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block with 3 variational parameters."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Layer of convolution blocks over all qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    return QuantumCircuit(num_qubits).append(qc_inst, qubits)

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def _pool_layer(sources: Iterable[int],
                sinks: Iterable[int],
                param_prefix: str) -> QuantumCircuit:
    """Pooling layer that pairs source and sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix,
                             length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(_pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    return QuantumCircuit(num_qubits).append(qc_inst, range(num_qubits))

# ----------------------------------------------------------------------
# QCNN construction
# ----------------------------------------------------------------------
def _build_qcnn(num_qubits: int = 8,
                conv_depth: int = 3,
                pool_depth: int = 3) -> EstimatorQNN:
    """Build a QCNN with a variational classifier on top."""
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits)

    # 1st convolution (full 8‑qubit layer)
    ansatz.compose(_conv_layer(num_qubits, "c1"),
                   list(range(num_qubits)), inplace=True)

    # 1st pooling (retain 4 qubits)
    ansatz.compose(_pool_layer(list(range(num_qubits // 2)),
                               list(range(num_qubits // 2, num_qubits)),
                               "p1"),
                   list(range(num_qubits)), inplace=True)

    # 2nd convolution on the remaining 4 qubits
    ansatz.compose(_conv_layer(num_qubits // 2, "c2"),
                   list(range(num_qubits // 2, num_qubits)), inplace=True)

    # 2nd pooling on the same 4 qubits
    ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "p2"),
                   list(range(num_qubits // 2, num_qubits)), inplace=True)

    # 3rd convolution on the last 2 qubits
    ansatz.compose(_conv_layer(num_qubits // 4, "c3"),
                   list(range(num_qubits - 2, num_qubits)), inplace=True)

    # 3rd pooling on the last 2 qubits
    ansatz.compose(_pool_layer([0], [1], "p3"),
                   list(range(num_qubits - 2, num_qubits)), inplace=True)

    # Variational classifier on the reduced qubit set
    classifier_circuit = _build_classifier_ansatz(num_qubits - 2, conv_depth)
    ansatz.compose(classifier_circuit,
                   list(range(num_qubits - 2, num_qubits)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observable: measure last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    return EstimatorQNN(circuit=circuit.decompose(),
                        observables=observable,
                        input_params=feature_map.parameters,
                        weight_params=ansatz.parameters,
                        estimator=Estimator())

# ----------------------------------------------------------------------
# Public hybrid QCNN wrapper
# ----------------------------------------------------------------------
class HybridQCNN:
    """Quantum wrapper around the hybrid QCNN.

    The class holds an EstimatorQNN instance and forwards calls to it,
    exposing a concise API that mirrors the classical counterpart.
    """
    def __init__(self, num_qubits: int = 8,
                 conv_depth: int = 3,
                 pool_depth: int = 3) -> None:
        self.qnn = _build_qcnn(num_qubits, conv_depth, pool_depth)

    def __call__(self, inputs):
        """Delegate evaluation to the underlying EstimatorQNN."""
        return self.qnn(inputs)

    @property
    def parameters(self):
        """Return the trainable parameters of the QCNN."""
        return self.qnn.weight_params

    @property
    def input_params(self):
        """Return the input parameters (feature map) of the QCNN."""
        return self.qnn.input_params

__all__ = ["HybridQCNN", "_build_qcnn"]
