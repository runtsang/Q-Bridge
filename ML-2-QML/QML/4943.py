import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Pauli
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.random import random_circuit

def ConvCircuit(kernel_size: int = 2, threshold: float = 0.0) -> QuantumCircuit:
    """Return a template quantum circuit used as a filter for 2×2 patches."""
    n_qubits = kernel_size ** 2
    qc = QuantumCircuit(n_qubits, name="ConvFilter")
    theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += random_circuit(n_qubits, 2, seed=42)
    qc.measure_all()
    return qc

def QCNNAnsatz(num_qubits: int = 8) -> QuantumCircuit:
    """Return the QCNN ansatz circuit as defined in the reference."""
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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = conv_circuit(params[param_index : param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = conv_circuit(params[param_index : param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            sub = pool_circuit(params[param_index : param_index + 3])
            qc.append(sub, [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    qc = QuantumCircuit(num_qubits, name="QCNN Ansatz")
    qc.append(conv_layer(num_qubits, "c1"), range(num_qubits))
    qc.append(pool_layer(range(num_qubits), range(num_qubits, 2 * num_qubits), "p1"), range(num_qubits))
    qc.append(conv_layer(num_qubits // 2, "c2"), range(num_qubits, 2 * num_qubits))
    qc.append(pool_layer(range(num_qubits, 2 * num_qubits), range(2 * num_qubits, 3 * num_qubits), "p2"), range(num_qubits, 2 * num_qubits))
    qc.append(conv_layer(num_qubits // 4, "c3"), range(2 * num_qubits, 3 * num_qubits))
    qc.append(pool_layer(range(2 * num_qubits), [2 * num_qubits + 1], "p3"), range(2 * num_qubits, 3 * num_qubits))
    return qc

def build_fraud_detection_qnn(
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit,
    shots: int = 1024,
) -> EstimatorQNN:
    """Construct a hybrid quantum neural network for fraud detection."""
    estimator = Estimator()
    observable = Pauli.from_label("Z" + "I" * (ansatz.num_qubits - 1))
    qnn = EstimatorQNN(
        circuit=ansatz,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        shots=shots,
    )
    return qnn

def build_fraud_detection_program(
    input_params,
    layers,
) -> EstimatorQNN:
    """
    Convenience wrapper that builds a QCNN-based fraud detection QNN.
    The feature map is a simple ZFeatureMap over the flattened input size.
    """
    input_dim = 4 * 14 * 14
    feature_map = ZFeatureMap(input_dim, reps=1, entanglement='full')
    ansatz = QCNNAnsatz(num_qubits=feature_map.num_qubits)
    return build_fraud_detection_qnn(feature_map, ansatz)

__all__ = [
    "ConvCircuit",
    "QCNNAnsatz",
    "build_fraud_detection_qnn",
    "build_fraud_detection_program",
]
