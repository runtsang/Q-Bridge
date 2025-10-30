import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNPlus:
    """
    Quantum convolutional neural network with parameter‑shared convolution
    blocks and adaptive readout. The topology mirrors the classical QCNNPlus
    but uses variational circuits to provide quantum expressivity.
    """
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.estimator = Estimator()
        self._build_circuit()

    def _build_circuit(self):
        # Feature map
        self.feature_map = ZFeatureMap(self.n_qubits)

        # Convolution block
        def conv_block(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1,0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0,1)
            qc.ry(params[2], 1)
            qc.cx(1,0)
            qc.rz(np.pi/2, 0)
            return qc

        # Convolution layer over all adjacent pairs
        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=(num_qubits//2)*3)
            idx = 0
            for i in range(0, num_qubits-1, 2):
                block = conv_block(params[idx:idx+3])
                qc.append(block, [i, i+1])
                idx += 3
            return qc

        # Pooling block
        def pool_block(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1,0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0,1)
            qc.ry(params[2], 1)
            return qc

        # Pooling layer over adjacent pairs
        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=(num_qubits//2)*3)
            idx = 0
            for i in range(0, num_qubits-1, 2):
                block = pool_block(params[idx:idx+3])
                qc.append(block, [i, i+1])
                idx += 3
            return qc

        # Hierarchical QCNN
        qc = QuantumCircuit(self.n_qubits)

        # Layer 1
        qc = qc.compose(conv_layer(self.n_qubits, "c1"))
        qc = qc.compose(pool_layer(self.n_qubits, "p1"))

        # Layer 2
        qc = qc.compose(conv_layer(self.n_qubits//2, "c2"))
        qc = qc.compose(pool_layer(self.n_qubits//2, "p2"))

        # Layer 3
        qc = qc.compose(conv_layer(self.n_qubits//4, "c3"))
        qc = qc.compose(pool_layer(self.n_qubits//4, "p3"))

        # Combine feature map and ansatz
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit = self.circuit.compose(self.feature_map, range(self.n_qubits))
        self.circuit = self.circuit.compose(qc, range(self.n_qubits))

        # Observable for binary classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits-1), 1)])

        # Build EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=qc.parameters,
            estimator=self.estimator,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the quantum neural network.
        X: array of shape (n_samples, n_qubits)
        """
        return self.qnn.predict(X)
