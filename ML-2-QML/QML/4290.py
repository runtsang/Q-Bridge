"""Hybrid quantum classifier that mirrors the classical helper.

The circuit combines a Z‑feature map, a random rotation layer, and a
convolution‑plus‑pooling stack inspired by the QCNN example.  It returns
the full variational circuit together with data‑encoding parameters,
trainable weight parameters and measurement observables.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def conv_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """One convolution block on a pair of qubits."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi/2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi/2, qubits[0])
    return qc

def pool_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """Pooling block that entangles two qubits and discards one."""
    qc = QuantumCircuit(len(qubits))
    qc.rz(-np.pi/2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Layer consisting of parallel convolution blocks."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        block = conv_circuit(params[param_index:param_index+3], [q1, q2])
        qc.append(block.to_instruction(), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Layer that reduces the qubit count by half."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        block = pool_circuit(params[param_index:param_index+3], [q1, q2])
        qc.append(block.to_instruction(), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a hybrid quantum circuit that mirrors the classical helper.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, typically equal to the number of input features.
    depth : int
        Number of convolution‑pooling repetitions.

    Returns
    -------
    circuit : QuantumCircuit
        The complete variational circuit.
    encoding : Iterable[ParameterVector]
        Parameter vectors that encode the classical data.
    weight_params : Iterable[ParameterVector]
        Trainable parameters of the ansatz.
    observables : list[SparsePauliOp]
        Observables measured at the end of the circuit.
    """
    # Feature‑map encoding
    feature_map = ZFeatureMap(num_qubits, reps=1)
    encoding = list(feature_map.parameters)

    # Ansatz: random layer + repeated conv‑pool blocks
    ansatz = QuantumCircuit(num_qubits)
    # Random layer for expressivity
    for q in range(num_qubits):
        ansatz.rx(np.pi/4, q)
        ansatz.rz(np.pi/3, q)
    # Convolution‑pooling stack
    current_qubits = num_qubits
    for d in range(depth):
        ansatz.append(conv_layer(current_qubits, f"c{d}").to_instruction(), range(current_qubits))
        # Pooling reduces qubits by half
        current_qubits = current_qubits // 2
        ansatz.append(pool_layer(current_qubits, f"p{d}").to_instruction(), range(current_qubits))

    weight_params = ansatz.parameters

    # Measure all qubits in Z
    observables = [SparsePauliOp("Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    # Assemble full circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.append(feature_map.to_instruction(), range(num_qubits))
    circuit.append(ansatz.to_instruction(), range(num_qubits))
    return circuit, encoding, weight_params, observables

__all__ = ["build_classifier_circuit"]
