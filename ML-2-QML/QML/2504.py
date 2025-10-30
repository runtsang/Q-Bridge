import numpy as np
import torch
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import Aer

class HybridQCNN:
    """Quantum‑classical hybrid QCNN that replaces the classical head with a variational circuit."""
    def __init__(self, feature_dim: int = 8, n_qubits: int = 8) -> None:
        self.feature_dim = feature_dim
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.estimator = Aer.get_backend("statevector_simulator")

        # Feature map
        self.feature_map = self._build_feature_map()

        # Ansatz
        self.ansatz = self._build_ansatz()

        # Full circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(self.feature_map, range(self.n_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(self.n_qubits), inplace=True)

        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        # EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_feature_map(self) -> QuantumCircuit:
        """Z‑feature map with 8 qubits."""
        params = ParameterVector("x", self.feature_dim)
        qc = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(params):
            qc.rz(p, i)
        return qc

    def _conv_block(self, params, q1, q2) -> QuantumCircuit:
        """Parametrised two‑qubit unitary used in convolution."""
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

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Two‑qubit convolution block applied pairwise."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}_c", num_qubits * 3)
        idx = 0
        for q in range(0, num_qubits, 2):
            sub = self._conv_block(params[idx:idx+3], q, q+1)
            qc.compose(sub, [q, q+1], inplace=True)
            idx += 3
        return qc

    def _pool_block(self, params, q1, q2) -> QuantumCircuit:
        """Parametrised two‑qubit unitary used in pooling."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        """Pooling block that reduces qubit count."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}_p", num_qubits // 2 * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            sub = self._pool_block(params[idx:idx+3], s, t)
            qc.compose(sub, [s, t], inplace=True)
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full QCNN ansatz with conv‑pool layers."""
        qc = QuantumCircuit(self.n_qubits)

        # First Convolutional Layer
        qc.compose(self._conv_layer(self.n_qubits, "c1"), range(self.n_qubits), inplace=True)

        # First Pooling Layer
        qc.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(self.n_qubits), inplace=True)

        # Second Convolutional Layer
        qc.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)

        # Second Pooling Layer
        qc.compose(self._pool_layer([0,1], [2,3], "p2"), range(4, 8), inplace=True)

        # Third Convolutional Layer
        qc.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)

        # Third Pooling Layer
        qc.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute quantum expectation values and return probabilities."""
        x_np = inputs.detach().cpu().numpy()
        out = self.qnn.predict(x_np)
        probs = 1 / (1 + np.exp(-out))  # sigmoid
        return torch.tensor(probs, dtype=torch.float32, device=inputs.device)

__all__ = ["HybridQCNN"]
