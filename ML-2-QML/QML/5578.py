import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuantumSelfAttention:
    """Quantum self‑attention subcircuit with rotation and controlled‑X entanglement."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params, entangle_params, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

class QuanvCircuit:
    """Quanvolution filter implemented as a random two‑qubit circuit."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: int):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class QCNNHybrid:
    """Quantum‑classical hybrid QCNN that mirrors the classical architecture."""
    def __init__(self):
        self.feature_map = ZFeatureMap(8)
        self.backend = Aer.get_backend("aer_simulator")
        self.shots = 512

        # Build the full ansatz
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, range(8), inplace=True)
        self.circuit.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        self.circuit.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8), inplace=True)
        self.circuit.compose(self._conv_layer(4, "c2"), range(4,8), inplace=True)
        self.circuit.compose(self._pool_layer([0,1], [2,3], "p2"), range(4,8), inplace=True)
        self.circuit.compose(self._conv_layer(2, "c3"), range(6,8), inplace=True)
        self.circuit.compose(self._pool_layer([0], [1], "p3"), range(6,8), inplace=True)

        # Self‑attention subcircuit
        self.circuit.compose(self._self_attention_circuit(4, "sa"), range(8), inplace=True)
        # Quanvolution filter
        self.circuit.compose(self._conv_filter_circuit(2, "vf"), range(4), inplace=True)

        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator
        )

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : (i + 1) * 3])
            qc.append(sub, [i, i + 1])
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for idx, (s, sk) in enumerate(zip(sources, sinks)):
            sub = self._pool_circuit(params[idx * 3 : (idx + 1) * 3])
            qc.append(sub, [s, sk])
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _self_attention_circuit(self, n_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector(prefix, length=n_qubits * 3)
        for i in range(n_qubits):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def _conv_filter_circuit(self, kernel_size: int, prefix: str) -> QuantumCircuit:
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector(prefix, length=n_qubits)
        for i in range(n_qubits):
            qc.rx(params[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the hybrid QCNN and return sigmoid probabilities."""
        expectations = self.qnn(inputs.tolist())
        probs = torch.sigmoid(torch.tensor(expectations, dtype=torch.float32))
        return probs

__all__ = ["QuantumSelfAttention", "QuanvCircuit", "QCNNHybrid"]
