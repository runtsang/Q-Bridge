import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQiskit
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in the QCNN ansatz."""
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
    """Two‑qubit pooling unitary used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer composed of pairwise conv_circuit blocks."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps a set of source qubits onto sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index+3]), [src, snk], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list, list, list[SparsePauliOp]]:
    """Simple layered ansatz with explicit encoding and variational parameters."""
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

def build_qcnn_circuit(num_qubits: int = 8, depth: int = 3) -> tuple[QuantumCircuit, list[SparsePauliOp]]:
    """Combines the feature map, convolution‑pooling ansatz and a classifier layer."""
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="QCNN Ansatz")

    # First convolution‑pooling cycle
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"),
                   range(num_qubits), inplace=True)

    # Subsequent cycles
    for d in range(2, depth + 1):
        in_qubits = num_qubits // (2 ** (d - 1))
        ansatz.compose(conv_layer(in_qubits, f"c{d}"), range(in_qubits), inplace=True)
        ansatz.compose(pool_layer(list(range(in_qubits // 2)),
                                 list(range(in_qubits // 2, in_qubits)),
                                 f"p{d}"), range(in_qubits), inplace=True)

    # Classifier ansatz appended after the pooling stages
    classifier_qc, _, _, obs = build_classifier_circuit(num_qubits, depth=1)
    ansatz.compose(classifier_qc, range(num_qubits), inplace=True)

    # Full circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    return circuit, obs

def QCNNGenQNN() -> EstimatorQNN:
    """Factory that returns an EstimatorQNN built from the hybrid QCNN ansatz."""
    estimator = EstimatorQiskit()
    circuit, observables = build_qcnn_circuit()
    feature_map = ZFeatureMap(8)
    # Decompose to avoid deep nesting
    circuit = circuit.decompose()
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["conv_circuit", "pool_circuit", "conv_layer", "pool_layer",
           "build_classifier_circuit", "build_qcnn_circuit", "QCNNGenQNN"]
