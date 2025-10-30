import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class QCNNHybridQNN:
    """
    Quantum feature extractor implementing a QCNN with convolution and pooling layers.
    Wrapped in a qiskit_machine_learning.neural_networks.EstimatorQNN for differentiable evaluation.
    """
    def __init__(self, num_qubits: int = 8, seed: int = 42) -> None:
        self.num_qubits = num_qubits
        self.estimator = StatevectorEstimator()
        self._circuit = self._build_circuit()
        self._qnn = QiskitEstimatorQNN(
            circuit=self._circuit.decompose(),
            observables=self._observable(),
            input_params=self._feature_map().parameters,
            weight_params=self._ansatz().parameters,
            estimator=self.estimator
        )

    def _feature_map(self) -> QuantumCircuit:
        return ZFeatureMap(self.num_qubits, reps=1, entanglement='circular')

    def _ansatz(self) -> QuantumCircuit:
        ans = QuantumCircuit(self.num_qubits)
        ans.compose(self._conv_layer(self.num_qubits, "c1"), list(range(self.num_qubits)), inplace=True)
        ans.compose(self._pool_layer(range(self.num_qubits//2), range(self.num_qubits//2, self.num_qubits), "p1"),
                     list(range(self.num_qubits)), inplace=True)
        ans.compose(self._conv_layer(self.num_qubits//2, "c2"), list(range(self.num_qubits//2, self.num_qubits)), inplace=True)
        ans.compose(self._pool_layer(range(self.num_qubits//4), range(self.num_qubits//4, self.num_qubits//2), "p2"),
                     list(range(self.num_qubits//2, self.num_qubits)), inplace=True)
        ans.compose(self._conv_layer(self.num_qubits//4, "c3"), list(range(self.num_qubits//4, self.num_qubits//2)), inplace=True)
        ans.compose(self._pool_layer([0], [1], "p3"),
                     list(range(self.num_qubits//4, self.num_qubits//2)), inplace=True)
        return ans

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            block = self._conv_block(params[i*3:(i+1)*3])
            qc.append(block, [i, i+1])
            qc.barrier()
        return qc

    def _conv_block(self, params):
        block = QuantumCircuit(2)
        block.rz(-np.pi/2, 1)
        block.cx(1, 0)
        block.rz(params[0], 0)
        block.ry(params[1], 1)
        block.cx(0, 1)
        block.ry(params[2], 1)
        block.cx(1, 0)
        block.rz(np.pi/2, 0)
        return block

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        num_pairs = len(sources)
        qc = QuantumCircuit(num_pairs * 2)
        params = ParameterVector(prefix, length=num_pairs * 3)
        for idx, (s, t) in enumerate(zip(sources, sinks)):
            block = self._pool_block(params[idx*3:(idx+1)*3])
            qc.append(block, [s, t])
            qc.barrier()
        return qc

    def _pool_block(self, params):
        block = QuantumCircuit(2)
        block.rz(-np.pi/2, 1)
        block.cx(1, 0)
        block.rz(params[0], 0)
        block.ry(params[1], 1)
        block.cx(0, 1)
        block.ry(params[2], 1)
        return block

    def _observable(self):
        return SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._qnn.predict(X)

def QCNNHybrid() -> QCNNHybridQNN:
    """Factory returning the configured QCNNHybridQNN."""
    return QCNNHybridQNN()

__all__ = ["QCNNHybrid", "QCNNHybridQNN"]
