import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class HybridQCNNModel(nn.Module):
    """
    Quantum‑classical hybrid QCNN that mirrors the classical pipeline:
      * A quantum feature map (ZFeatureMap).
      * Variational convolution and pooling layers implemented with Qiskit.
      * A StatevectorEstimator‑based EstimatorQNN for quantum circuit evaluation.
      * A classical linear head that incorporates an RBF kernel on the quantum output.
    """
    def __init__(self, num_qubits: int = 8, gamma: float = 1.0, seed: int = 12345):
        super().__init__()
        algorithm_globals.random_seed = seed
        self.estimator = StatevectorEstimator()
        # Classical feature map
        self.feature_map = ZFeatureMap(num_qubits)
        # Quantum ansatz
        self.ansatz = self._build_ansatz(num_qubits)
        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        # Quantum neural network
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
        # Kernel branch
        self.gamma = gamma
        self.prototype = nn.Parameter(torch.randn(1))
        # Classical head
        self.head = nn.Linear(2, 1)

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Constructs the variational QCNN ansatz using convolution and pooling layers."""
        def conv_circuit(params):
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, qubits)
            return qc

        def pool_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            return target

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, range(num_qubits))
            return qc

        # Build the full ansatz
        ansatz = QuantumCircuit(num_qubits, name="Ansatz")
        ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(num_qubits), inplace=True)
        ansatz.compose(conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2, num_qubits), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(num_qubits // 2, num_qubits), inplace=True)
        ansatz.compose(conv_layer(num_qubits // 4, "c3"), range(num_qubits - 2, num_qubits), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(num_qubits - 2, num_qubits), inplace=True)
        return ansatz

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for the quantum circuit
        inputs_np = inputs.cpu().detach().numpy()
        qnn_output = self.qnn(inputs_np)  # shape (batch, 1)
        qnn_tensor = torch.tensor(qnn_output, device=inputs.device, dtype=torch.float32)
        # Classical kernel on quantum output
        k = torch.exp(-self.gamma * torch.sum((qnn_tensor - self.prototype) ** 2, dim=-1, keepdim=True))
        x = torch.cat([qnn_tensor, k], dim=-1)  # (batch, 2)
        return self.head(x).squeeze(-1)


__all__ = ["HybridQCNNModel"]
