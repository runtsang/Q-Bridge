import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import SparsePauliOp

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Convolutional block with a small random layer followed by a parameterized kernel."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(num_qubits):
        qc.ry(params[i], i)
        qc.rz(params[num_qubits + i], i)
        qc.rx(params[2 * num_qubits + i], i)
    # random mixing
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc

def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Pooling block that entangles pairs of qubits and applies rotations."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 2)
    for i in range(0, num_qubits, 2):
        qc.cx(i, i + 1)
        qc.rz(params[i], i)
        qc.ry(params[i + 1], i + 1)
    return qc

def QCNN_QML() -> SamplerQNN:
    """Hybrid quantum QCNN that outputs a probability distribution."""
    # Feature map for the input data
    feature_map = ZFeatureMap(8)
    # Build ansatz with convolution and pooling layers
    ansatz = QuantumCircuit(8)
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), inplace=True)
    # Combine feature map and ansatz into a single circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    # Sampler for the final measurement
    sampler = StatevectorSampler()
    sampler_qnn = SamplerQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler
    )
    return sampler_qnn

__all__ = ["QCNN_QML"]
