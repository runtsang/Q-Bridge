import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QuantumHybridCNN(nn.Module):
    """
    QCNN‑style quantum circuit wrapped in an EstimatorQNN.
    The circuit consists of a ZFeatureMap followed by convolutional and pooling layers.
    """

    def __init__(self, input_dim: int = 8, shots: int = 100) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.shots = shots

        # Feature map for input encoding
        self.feature_map = ZFeatureMap(self.input_dim)

        # Build the QCNN ansatz
        self.ansatz = self._build_ansatz()

        # Estimator for expectation evaluation
        self.estimator = StatevectorEstimator()

        # QNN wrapping the circuit
        self.qnn = EstimatorQNN(
            circuit=self.full_circuit().decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (self.input_dim - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_idx = 0
        params = ParameterVector(prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[param_idx:param_idx + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(self._conv_circuit(params[param_idx:param_idx + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_idx += 3
        return qc

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_idx = 0
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[param_idx:param_idx + 3]), [src, snk], inplace=True)
            qc.barrier()
            param_idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.input_dim)

        # First convolutional layer
        ansatz.compose(self._conv_layer(self.input_dim, "c1"), inplace=True)

        # First pooling layer
        ansatz.compose(
            self._pool_layer(
                range(self.input_dim // 2),
                range(self.input_dim // 2, self.input_dim),
                "p1",
            ),
            inplace=True,
        )

        # Additional layers could be appended here in the same style
        return ansatz

    def full_circuit(self) -> QuantumCircuit:
        """Full circuit: feature map + ansatz."""
        circ = QuantumCircuit(self.input_dim)
        circ.compose(self.feature_map, inplace=True)
        circ.compose(self.ansatz, inplace=True)
        return circ

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs the EstimatorQNN on the batch of inputs.
        Returns a probability vector of shape (batch, 2) for binary classification.
        """
        # Convert to NumPy array for the QNN
        x_np = inputs.detach().cpu().numpy()

        # Expectation values in [-1, 1] -> convert to [0, 1]
        exp_vals = self.qnn(x_np).reshape(-1, 1)
        probs = (exp_vals + 1) / 2

        probs = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumHybridCNN"]
