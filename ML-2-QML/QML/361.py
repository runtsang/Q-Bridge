import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import L_BFGS_B
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

class QCNN:
    """
    Quantum convolution‑pooling neural network.

    Features
    -------
    * Depth‑controlled convolution and pooling layers.
    * Multi‑observable readout (Z on each qubit).
    * Parameter‑shift gradient estimation via EstimatorQNN.
    * Lightweight ``train`` method using L_BFGS_B optimizer.

    Parameters
    ----------
    num_qubits : int, default 8
        Total number of qubits for the feature map and ansatz.
    conv_depth : int, default 3
        Number of convolutional layers.
    pool_depth : int, default 3
        Number of pooling layers.
    """
    def __init__(self,
                 num_qubits: int = 8,
                 conv_depth: int = 3,
                 pool_depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.conv_depth = conv_depth
        self.pool_depth = pool_depth
        self.feature_map = ZFeatureMap(num_qubits)
        self.circuit = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(circuit=self.circuit.decompose(),
                                observables=self.observable,
                                input_params=self.feature_map.parameters,
                                weight_params=self.circuit.parameters,
                                estimator=self.estimator)

    def _conv_layer(self, qubits: list[int], prefix: str) -> QuantumCircuit:
        """Return a convolution block acting on adjacent qubit pairs."""
        qc = QuantumCircuit(len(qubits), name="conv")
        params = ParameterVector(prefix, length=len(qubits) * 3)
        idx = 0
        for i in range(0, len(qubits) - 1, 2):
            sub = self._conv_unit(params[idx:idx+3], qubits[i], qubits[i+1])
            qc.append(sub, [qubits[i], qubits[i+1]])
            idx += 3
        return qc

    def _conv_unit(self, params, q1: int, q2: int) -> QuantumCircuit:
        """Single 2‑qubit convolution unit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, q2)
        sub.cx(q2, q1)
        sub.rz(params[0], q1)
        sub.ry(params[1], q2)
        sub.cx(q1, q2)
        sub.ry(params[2], q2)
        sub.cx(q2, q1)
        sub.rz(np.pi/2, q1)
        return sub

    def _pool_layer(self, qubits: list[int], prefix: str) -> QuantumCircuit:
        """Return a pooling block that annihilates half the qubits."""
        qc = QuantumCircuit(len(qubits), name="pool")
        params = ParameterVector(prefix, length=len(qubits)//2 * 3)
        idx = 0
        for i in range(0, len(qubits) - 1, 2):
            sub = self._pool_unit(params[idx:idx+3], qubits[i], qubits[i+1])
            qc.append(sub, [qubits[i], qubits[i+1]])
            idx += 3
        return qc

    def _pool_unit(self, params, q1: int, q2: int) -> QuantumCircuit:
        """Single 2‑qubit pooling unit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi/2, q2)
        sub.cx(q2, q1)
        sub.rz(params[0], q1)
        sub.ry(params[1], q2)
        sub.cx(q1, q2)
        sub.ry(params[2], q2)
        return sub

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full ansatz with alternating conv/pool layers."""
        qc = QuantumCircuit(self.num_qubits)
        # Feature map
        qc.append(self.feature_map, range(self.num_qubits))
        # Convolution‑pooling stages
        qubits = list(range(self.num_qubits))
        for d in range(self.conv_depth):
            qc.append(self._conv_layer(qubits, f"c{d}"), qubits)
            if d < self.pool_depth:
                qc.append(self._pool_layer(qubits, f"p{d}"), qubits)
                # Reduce qubit count by discarding every second qubit
                qubits = qubits[::2]
        return qc

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 50,
              learning_rate: float = 0.1) -> None:
        """
        Train the QNN with a classical optimizer.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        y : np.ndarray
            Binary labels, shape (n_samples, ).
        epochs : int
            Number of optimisation iterations.
        learning_rate : float
            Step size for the optimizer.
        """
        clf = NeuralNetworkClassifier(
            estimator=self.qnn,
            optimizer=L_BFGS_B(maxiter=epochs, maxcor=10),
            loss='hinge',
            num_classes=2,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=0
        )
        clf.fit(X, y)
        self.trained_params = clf.get_params()['estimator'].parameters

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        probs = self.qnn.predict(X)
        return (probs > 0.5).astype(int)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, " \
               f"conv_depth={self.conv_depth}, pool_depth={self.pool_depth})"
