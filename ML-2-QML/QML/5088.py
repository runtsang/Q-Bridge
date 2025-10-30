from __future__ import annotations

from typing import List
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

def _convolution_layer(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
    """QCNN convolution block operating on qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        qc.ry(params[i], i)
        qc.ry(params[i+1], i+1)
        qc.cx(i, i+1)
    return qc

def _pooling_layer(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
    """QCNN pooling block that reduces qubits by half."""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        qc.cx(i, i+1)
        qc.ry(params[i], i)
        qc.ry(params[i+1], i+1)
    return qc

def build_qcnn_circuit(num_qubits: int = 8) -> QuantumCircuit:
    """Assemble a QCNN‑style variational circuit."""
    total_params = num_qubits * 3          # 3 params per qubit per layer
    params = ParameterVector("θ", total_params)
    qc = QuantumCircuit(num_qubits)
    # Feature map
    feature_map = ZZFeatureMap(num_qubits, reps=1)
    qc.compose(feature_map, inplace=True)
    # First convolution + pooling on full register
    conv1 = _convolution_layer(num_qubits, params[0:num_qubits*3:3])
    pool1 = _pooling_layer(num_qubits, params[num_qubits*3:2*num_qubits*3:3])
    qc.compose(conv1, inplace=True)
    qc.compose(pool1, inplace=True)
    # Second convolution + pooling on reduced register
    conv2 = _convolution_layer(num_qubits // 2, params[2*num_qubits*3:3*num_qubits*3:3])
    pool2 = _pooling_layer(num_qubits // 2, params[3*num_qubits*3:4*num_qubits*3:3])
    qc.compose(conv2, inplace=True)
    qc.compose(pool2, inplace=True)
    return qc

def hybrid_sampler_qnn() -> SamplerQNN:
    """Return a Qiskit MachineLearning SamplerQNN that uses QCNN layers."""
    circuit = build_qcnn_circuit()
    weight_params = circuit.parameters
    sampler = StatevectorSampler()
    return SamplerQNN(circuit=circuit,
                      input_params=[],
                      weight_params=weight_params,
                      sampler=sampler)

def swap_test_overlap(state1: Statevector, state2: Statevector) -> float:
    """Estimate the overlap of two statevectors via a swap test."""
    swap_circuit = QuantumCircuit(3, 1)
    swap_circuit.h(0)
    swap_circuit.cx(0, 1)
    swap_circuit.cx(0, 2)
    swap_circuit.h(0)
    # Append state1 and state2 to qubits 1 and 2
    swap_circuit.append(state1.data, [1])
    swap_circuit.append(state2.data, [2])
    swap_circuit.measure(0, 0)
    backend = Aer.get_backend('qasm_simulator')
    result = execute(swap_circuit, backend, shots=1024).result()
    counts = result.get_counts()
    return (counts.get('0', 0) - counts.get('1', 0)) / 1024

__all__ = ["hybrid_sampler_qnn", "swap_test_overlap"]
