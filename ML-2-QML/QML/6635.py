import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNQuanvolutionHybrid:
    """Quantum implementation of the QCNN‑Quanvolution hybrid.
    Builds a QCNN ansatz with convolutional and pooling layers and
    a ZFeatureMap.  The model can be used as a QNN with a state‑vector
    estimator.  The architecture mirrors the classical version but
    operates on quantum states.
    """

    def __init__(self, num_qubits: int = 8, num_classes: int = 10):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.qnn = self._build_qnn()

    def _conv_circuit(self, params):
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

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc.append(self._pool_circuit(params[param_index:param_index+3]), [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    def _build_ansatz(self):
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)

        # First Pooling Layer
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

        # Second Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)

        # Second Pooling Layer
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)

        # Third Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"), inplace=True)

        # Third Pooling Layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        return ansatz

    def _build_qnn(self):
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = self._build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        estimator = StatevectorEstimator()
        qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        return qnn

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Run the QNN on a batch of binary feature vectors.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, num_qubits).  Each row is a binary feature vector.

        Returns
        -------
        np.ndarray
            Raw expectation values of shape (batch, 1).
        """
        return self.qnn.predict(inputs).reshape(-1, 1)

__all__ = ["QCNNQuanvolutionHybrid"]
