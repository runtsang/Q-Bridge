import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import Aer
from qiskit.primitives import Estimator as EstimatorPrimitive

class QuantumExpectationLayer:
    """Wrapper that evaluates a parameterised circuit and returns the Z expectation."""
    def __init__(self, num_qubits: int, backend=None, shots: int = 100):
        self.num_qubits = num_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.h(range(num_qubits))
        self.theta = ParameterVector("theta", length=num_qubits)
        self.circuit.ry(self.theta, range(num_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta[i]: theta for i, theta in enumerate(thetas)}])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        probs = np.array(list(result.values())) / self.shots
        states = np.array([int(k, 2) for k in result.keys()])
        return np.array([np.sum(states * probs)])

class QCNNHybridGen:
    """
    Quantum QCNN that mirrors the classical dense layers with a quantum convolutional ansatz
    and a quantum expectation head. The network can be used purely for inference or
    as a differentiable layer in a hybrid training loop.
    """
    def __init__(self,
                 num_qubits: int = 8,
                 backend=None,
                 shots: int = 100,
                 shift: float = np.pi / 2):
        self.num_qubits = num_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.shift = shift
        self.feature_map = QuantumExpectationLayer(num_qubits, self.backend, shots)
        self.ansatz = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        algorithm_globals.estimator = EstimatorPrimitive()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.circuit.parameters,
            weight_params=self.ansatz.parameters,
            estimator=algorithm_globals.estimator
        )

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

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i:i+3])
            qc.append(sub, [i, i+1])
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(prefix, length=(len(sources) // 2) * 3)
        for src, sink, p in zip(sources, sinks, params):
            sub = QuantumCircuit(2)
            sub.rz(-np.pi / 2, 1)
            sub.cx(1, 0)
            sub.rz(p[0], 0)
            sub.ry(p[1], 1)
            sub.cx(0, 1)
            sub.ry(p[2], 1)
            qc.append(sub, [src, sink])
        return qc

    def _build_ansatz(self):
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        qc.compose(self._pool_layer(list(range(self.num_qubits//2)),
                                    list(range(self.num_qubits//2, self.num_qubits)),
                                    "p1"), inplace=True)
        qc.compose(self._conv_layer(self.num_qubits//2, "c2"), inplace=True)
        qc.compose(self._pool_layer(list(range(self.num_qubits//4)),
                                    list(range(self.num_qubits//4, self.num_qubits//2)),
                                    "p2"), inplace=True)
        qc.compose(self._conv_layer(self.num_qubits//4, "c3"), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return qc

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: x is a 2‑D array of shape (batch, num_qubits).
        Returns the expectation values from the quantum circuit.
        """
        return self.qnn.predict(x)

def QCNN() -> QCNNHybridGen:
    """Factory for the quantum hybrid QCNN."""
    return QCNNHybridGen()

__all__ = ["QCNNHybridGen", "QCNN"]
