"""Quantum circuit factories for hybrid classification experiments.

The module provides a generic feature‑encoding circuit, a QCNN‑style ansatz,
and a helper to wrap a circuit into a Qiskit EstimatorQNN.  All functions
return the objects needed for end‑to‑end training with the Qiskit ML stack.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a layered ansatz with data‑encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables

def build_qcnn_circuit(num_qubits: int) -> QuantumCircuit:
    """Build a QCNN‑style circuit with convolution and pooling layers.

    Parameters
    ----------
    num_qubits : int
        Size of the QCNN (must be a multiple of 2).
    """
    # Convolution block for two qubits
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_idx = 0
        params = ParameterVector(prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(conv_circuit(params[param_idx:param_idx+3]), [q1, q2])
            qc.barrier()
            param_idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(conv_circuit(params[param_idx:param_idx+3]), [q1, q2])
            qc.barrier()
            param_idx += 3
        return qc

    def pool_layer(sources: List[int], sinks: List[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_idx = 0
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for src, snk in zip(sources, sinks):
            qc.append(pool_circuit(params[param_idx:param_idx+3]), [src, snk])
            qc.barrier()
            param_idx += 3
        return qc

    # Build the full 8‑qubit QCNN ansatz (simplified for arbitrary size)
    qc = QuantumCircuit(num_qubits)
    qc.compose(conv_layer(num_qubits, "c1"), inplace=True)
    qc.compose(pool_layer(list(range(num_qubits)), list(range(num_qubits, 2*num_qubits)), "p1"), inplace=True)
    return qc

def build_estimator_qnn(circuit: QuantumCircuit, feature_map: QuantumCircuit) -> EstimatorQNN:
    """Wrap a QCNN circuit into a Qiskit EstimatorQNN.

    Parameters
    ----------
    circuit : QuantumCircuit
        The QCNN ansatz.
    feature_map : QuantumCircuit
        The classical data‑encoding circuit.
    """
    estimator = StatevectorEstimator()
    observable = SparsePauliOp.from_list([("Z" + "I" * (circuit.num_qubits - 1), 1)])
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = [
    "build_classifier_circuit",
    "build_qcnn_circuit",
    "build_estimator_qnn",
]
