import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import Iterable, Tuple, List

def conv_circuit(params):
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

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

class HybridClassifier:
    """Hybrid quantum classifier that can compose convolution, pooling, and optional LSTM-inspired layers."""
    def __init__(self, num_qubits: int, depth: int = 3, use_lstm: bool = False, use_cnn: bool = True) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth, use_lstm, use_cnn)

    def get_qnn(self):
        estimator = StatevectorEstimator()
        qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=estimator,
        )
        return qnn

def build_classifier_circuit(num_qubits: int, depth: int, use_lstm: bool = False, use_cnn: bool = True) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    if use_cnn:
        for d in range(depth):
            circuit = circuit.compose(conv_layer(num_qubits, f"c{d}"))
    else:
        for i in range(num_qubits):
            circuit.ry(weights[i], i)
    observables = [SparsePauliOp(f"Z{'I'*(num_qubits-1)}")]
    return circuit, list(encoding), list(weights), observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
