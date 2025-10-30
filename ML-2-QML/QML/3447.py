import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class UnifiedQCNN:
    """
    Quantum QCNN that accepts an 8‑dimensional classical feature vector.
    The circuit reproduces the convolutional and pooling layers of the
    original QCNN, while the feature map is fed with the classical
    representation.  The class exposes a ``predict`` method compatible
    with the classical backbone.
    """
    def __init__(self) -> None:
        self.estimator = Estimator()
        feature_map = ZFeatureMap(8)
        ansatz = self._build_ansatz()
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the convolution–pooling ansatz."""
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

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            params = ParameterVector(prefix, length=num_qubits * 3)
            for idx in range(0, num_qubits, 2):
                qc.append(conv_circuit(params[idx*3:(idx+1)*3]), [idx, idx+1])
                qc.barrier()
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

        def pool_layer(sources, sinks, prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            params = ParameterVector(prefix, length=num_qubits // 2 * 3)
            for i, (src, sink) in enumerate(zip(sources, sinks)):
                qc.append(pool_circuit(params[i*3:(i+1)*3]), [src, sink])
                qc.barrier()
            return qc

        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), range(4,8), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], "p2"), range(4,8), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), range(6,8), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(6,8), inplace=True)
        return ansatz

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.  ``x`` should be an array of shape (batch, 8).
        Returns the QCNN output probabilities.
        """
        return self.qnn.predict(x)

def create_unified_qcnn() -> UnifiedQCNN:
    """Factory returning the configured :class:`UnifiedQCNN`."""
    return UnifiedQCNN()

__all__ = ["UnifiedQCNN", "create_unified_qcnn"]
