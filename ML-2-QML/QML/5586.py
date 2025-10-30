"""Quantum hybrid CNN classifier.

The quantum circuit implements a QCNN‑style convolutional and pooling layer,
followed by a QFC‑style random layer and a quantum classifier head.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a quantum classifier circuit that mirrors the hybrid architecture:
    - data encoding with a Z‑style feature map
    - QCNN convolution + pooling layers with variational parameters
    - QFC‑style random layer and measurement
    - Final measurement observables
    """
    # Feature map – simple X‑rotation encoding
    feature_map = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        feature_map.h(i)
        feature_map.rz(ParameterVector(f"x_{i}", 1)[0], i)

    # Variational parameters for convolution and pooling
    conv_params = ParameterVector("c", length=num_qubits * 3)
    pool_params = ParameterVector("p", length=num_qubits // 2 * 3)

    # QCNN convolutional layer
    conv_layer = QuantumCircuit(num_qubits)
    for q in range(0, num_qubits, 2):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(conv_params[q], 0)
        qc.ry(conv_params[q+1], 1)
        qc.cx(0, 1)
        qc.ry(conv_params[q+2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi/2, 0)
        conv_layer.append(qc, [q, q+1])
    conv_layer.barrier()

    # QCNN pooling layer
    pool_layer = QuantumCircuit(num_qubits)
    for q in range(0, num_qubits, 2):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(pool_params[q//2], 0)
        qc.ry(pool_params[q//2+1], 1)
        qc.cx(0, 1)
        qc.ry(pool_params[q//2+2], 1)
        pool_layer.append(qc, [q, q+1])
    pool_layer.barrier()

    # QFC‑style random layer
    rand_layer = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for q in range(num_qubits):
            rand_layer.ry(ParameterVector(f"theta_{q}_{_}", 1)[0], q)
        for q in range(num_qubits-1):
            rand_layer.cz(q, q+1)

    # Assemble full circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(conv_layer, inplace=True)
    circuit.compose(pool_layer, inplace=True)
    circuit.compose(rand_layer, inplace=True)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I"*i+"Z"+"I"*(num_qubits-i-1)) for i in range(num_qubits)]
    return circuit, list(feature_map.parameters), list(rand_layer.parameters), observables

class QuantumHybridCNN:
    """
    Quantum implementation of the hybrid CNN architecture.
    """
    def __init__(self, num_qubits: int = 8, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.weight_params, self.observables = build_classifier_circuit(num_qubits, depth)

    def evaluate(self, data: np.ndarray):
        """
        Evaluate the circuit on classical data. Data should be a 1D array of length num_qubits.
        """
        from qiskit import Aer, execute
        backend = Aer.get_backend("statevector_simulator")
        bound_circuit = self.circuit.bind_parameters({p: val for p, val in zip(self.encoding_params, data)})
        job = execute(bound_circuit, backend)
        result = job.result()
        statevector = result.get_statevector(bound_circuit)
        return statevector

__all__ = ["build_classifier_circuit", "QuantumHybridCNN"]
