"""Hybrid quantum sampler combining QCNN-style ansatz with statevector sampling."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def HybridSamplerQNN():
    """Factory returning a hybrid quantum sampler."""
    # Feature map
    feature_map = ZFeatureMap(8)

    # Convolution circuit
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

    # Convolution layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    # Pooling circuit
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Pooling layer
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc.compose(pool_circuit(params[param_index:param_index + 3]), [source, sink], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    # Build ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Sampler
    sampler = StatevectorSampler()
    sampler_qnn = SamplerQNN(
        circuit=circuit.decompose(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler,
    )
    return sampler_qnn


__all__ = ["HybridSamplerQNN"]
