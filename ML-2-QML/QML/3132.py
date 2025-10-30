from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution unit used in the QCNN ansatz.
    """
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
    """
    Builds a convolutional layer that applies _conv_circuit on adjacent qubit pairs.
    """
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
    """
    Two‑qubit pooling unit used in the QCNN ansatz.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Builds a pooling layer that applies _pool_circuit on pairs of qubits.
    """
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(range(num_qubits // 2), range(num_qubits // 2, num_qubits)):
        qc.append(_pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc


def HybridSamplerQNN() -> SamplerQNN:
    """
    Constructs a quantum sampler that mirrors the classical HybridSamplerCNN architecture.
    The feature map prepares the input encoding, the ansatz implements three
    convolution–pooling stages, and a StatevectorSampler returns a probability
    distribution over two classical outcomes.
    """
    # Feature map for 8 qubits
    feature_map = ZFeatureMap(8)

    # Build the ansatz with three conv–pool stages
    ansatz = QuantumCircuit(8, name="Hybrid QCNN Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), range(8), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), range(4, 8), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable for a two‑outcome measurement (Z on qubit 0)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Statevector sampler to produce probability distribution
    sampler = StatevectorSampler()

    # Construct the SamplerQNN
    return SamplerQNN(
        circuit=circuit.decompose(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler,
        observables=observable,
    )


__all__ = ["HybridSamplerQNN", "HybridSamplerQNN"]
