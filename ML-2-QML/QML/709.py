import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNModel(torch.nn.Module):
    """Quantum Convolutional Neural Network with a variational ansatz."""

    def __init__(self, seed: int = 12345) -> None:
        super().__init__()
        self.seed = seed
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
        # Expose quantum parameters as torch parameters
        self.weight_params = torch.nn.Parameter(
            torch.tensor([p.value() for p in self.ansatz.parameters], dtype=torch.float32)
        )

    def _build_ansatz(self) -> QuantumCircuit:
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

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits * 3)
            for i in range(0, num_qubits, 2):
                qc.append(conv_circuit(params[i * 3 : (i + 1) * 3]), [i, i + 1])
                qc.barrier()
            return qc

        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
            for i in range(0, num_qubits, 2):
                qc.append(pool_circuit(params[(i // 2) * 3 : (i // 2 + 1) * 3]), [i, i + 1])
                qc.barrier()
            return qc

        qc = QuantumCircuit(8)
        qc.compose(conv_layer(8, "c1"), inplace=True)
        qc.compose(pool_layer(8, "p1"), inplace=True)
        qc.compose(conv_layer(4, "c2"), inplace=True)
        qc.compose(pool_layer(4, "p2"), inplace=True)
        qc.compose(conv_layer(2, "c3"), inplace=True)
        qc.compose(pool_layer(2, "p3"), inplace=True)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using the quantum circuit."""
        inputs = x.detach().cpu().numpy()
        preds = self.qnn.predict(inputs)
        return torch.tensor(preds, dtype=torch.float32, device=x.device)

    def set_params(self, params: torch.Tensor) -> None:
        """Update the variational parameters of the ansatz."""
        for p, val in zip(self.ansatz.parameters, params.detach().cpu().numpy()):
            p.assign(val)

    def get_params(self) -> torch.Tensor:
        """Retrieve current variational parameters."""
        return torch.tensor([p.value() for p in self.ansatz.parameters], dtype=torch.float32)
