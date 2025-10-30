import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
    """Two‑qubit convolution unitary used in QCNN."""
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

def _pool_circuit(params):
    """Two‑qubit pooling unitary used in QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, param_prefix):
    """Assemble a convolutional layer as a parameter‑shared instruction."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _pool_layer(sources, sinks, param_prefix):
    """Assemble a pooling layer as a parameter‑shared instruction."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, snk in zip(sources, sinks):
        qc.append(_pool_circuit(params[param_index:param_index+3]), [src, snk])
        qc.barrier()
        param_index += 3
    return qc

class QuanvolutionHybridQNN:
    """QCNN‑style variational quantum neural network that encodes image
    patches, applies convolutional and pooling layers, and returns class
    probabilities. Mirrors the structure used in the QCNN reference."""
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.estimator = Estimator()
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        # Feature map transforms classical data into quantum states
        feature_map = ZFeatureMap(self.n_qubits)
        # Build ansatz with convolution and pooling layers
        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        # First convolution & pooling
        ansatz.compose(_conv_layer(self.n_qubits, "c1"), inplace=True)
        ansatz.compose(_pool_layer(range(self.n_qubits//2), range(self.n_qubits//2, self.n_qubits), "p1"), inplace=True)
        # Second convolution & pooling
        ansatz.compose(_conv_layer(self.n_qubits//2, "c2"), inplace=True)
        ansatz.compose(_pool_layer(range(self.n_qubits//4), range(self.n_qubits//4, self.n_qubits//2), "p2"), inplace=True)
        # Final measurement observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits-1), 1)])
        # Combine feature map and ansatz into a single circuit
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        # Wrap into EstimatorQNN
        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the given batch of inputs."""
        return self.qnn.predict(inputs)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.predict(inputs)

__all__ = ["QuanvolutionHybridQNN"]
