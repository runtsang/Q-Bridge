import numpy as np
import qiskit
from qiskit import QuantumCircuit, ParameterVector, transpile, assemble, Aer
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

class QCNNQuantumHead:
    """Quantum head implementing a QCNN ansatz and returning the expectation of a Pauli‑Z operator."""
    def __init__(self,
                 num_qubits: int = 8,
                 backend=None,
                 shots: int = 1024):
        self.num_qubits = num_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.feature_map = ZFeatureMap(num_qubits)
        self.circuit = self._build_ansatz()
        self.estimator = Estimator()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.weight_params = list(self.circuit.parameters)

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(self._conv_circuit(params[idx:idx+3]), [q1, q2])
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            qc.append(self._pool_circuit(params[idx:idx+3]), [src, snk])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.num_qubits)
        # First convolutional layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        # First pooling layer
        ansatz.compose(self._pool_layer(list(range(self.num_qubits//2)), list(range(self.num_qubits//2, self.num_qubits)), "p1"), inplace=True)
        # Second convolutional layer
        ansatz.compose(self._conv_layer(self.num_qubits//2, "c2"), inplace=True)
        # Second pooling layer
        ansatz.compose(self._pool_layer(list(range(self.num_qubits//4)), list(range(self.num_qubits//4, self.num_qubits//2)), "p2"), inplace=True)
        # Third convolutional layer
        ansatz.compose(self._conv_layer(self.num_qubits//4, "c3"), inplace=True)
        # Third pooling layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return ansatz

    def evaluate(self, input_data: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of input data."""
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        full_circuit.compose(self.circuit, range(self.num_qubits), inplace=True)
        full_circuit = full_circuit.decompose()
        results = self.estimator.run(
            circuits=[full_circuit],
            parameter_binds=[dict(zip(self.feature_map.parameters, x)) for x in input_data],
            observables=[self.observable],
            backend=self.backend,
        )
        expectations = np.array([res.data['expectation_value'] for res in results])
        return expectations

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return self.evaluate(input_data)

__all__ = ["QCNNQuantumHead"]
