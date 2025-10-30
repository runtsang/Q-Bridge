import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNHybridQNN:
    """Quantum hybrid classifier mirroring the classical QCNNHybrid architecture."""
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(n_qubits)
        self.circuit = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator
        )

    def _conv_circuit(self, params, q1, q2):
        c = QuantumCircuit(2)
        c.rz(-np.pi/2, 1)
        c.cx(1,0)
        c.rz(params[0],0)
        c.ry(params[1],1)
        c.cx(0,1)
        c.ry(params[2],1)
        c.cx(1,0)
        c.rz(np.pi/2,0)
        return c

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
            sub = self._conv_circuit(params[3*i:3*i+3], q1, q2)
            qc.append(sub, [q1, q2])
            qc.barrier()
        return qc

    def _pool_circuit(self, params, src, snk):
        c = QuantumCircuit(2)
        c.rz(-np.pi/2, 1)
        c.cx(1,0)
        c.rz(params[0],0)
        c.ry(params[1],1)
        c.cx(0,1)
        c.ry(params[2],1)
        return c

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
        for i, (src, snk) in enumerate(zip(sources, sinks)):
            sub = self._pool_circuit(params[3*i:3*i+3], src, snk)
            qc.append(sub, [src, snk])
            qc.barrier()
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ans = QuantumCircuit(self.n_qubits, name="Ansatz")
        ans.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        ans.compose(self._pool_layer([0,1,2,3],[4,5,6,7],"p1"), inplace=True)
        ans.compose(self._conv_layer(4, "c2"), inplace=True)
        ans.compose(self._pool_layer([0,1],[2,3],"p2"), inplace=True)
        ans.compose(self._conv_layer(2, "c3"), inplace=True)
        ans.compose(self._pool_layer([0],[1],"p3"), inplace=True)
        return ans

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities for binary classification."""
        return self.qnn.predict(X)

def QCNNHybridQNNFactory() -> QCNNHybridQNN:
    """Factory returning a configured :class:`QCNNHybridQNN`."""
    return QCNNHybridQNN()
