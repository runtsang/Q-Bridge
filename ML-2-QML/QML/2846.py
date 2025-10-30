import torch
import torch.nn as nn
import torchquantum as tq
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumFilter(tq.QuantumModule):
    """Parameterized quantum kernel that operates on 2‑qubit patches."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        # Simple Ry encoding for each input qubit
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.output_dim = n_wires
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, n_wires * n_patches)
        returns: tensor of shape (batch, n_wires * n_patches)
        """
        bsz = x.size(0)
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        n_patches = x.size(1) // self.n_wires
        for p in range(n_patches):
            data = x[:, p*self.n_wires : (p+1)*self.n_wires]
            self.encoder(qdev, data)
            self.random_layer(qdev)
            measurement = self.measure(qdev)
            patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

class EstimatorQNNWrapper(nn.Module):
    """EstimatorQNN from Qiskit that maps a vector of qubit angles to an expectation value."""
    def __init__(self, n_qubits: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.input_params = [Parameter(f"in_{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, n_qubits)
        returns: tensor of shape (batch, 1)
        """
        bsz = x.size(0)
        out = []
        for i in range(bsz):
            params = {self.input_params[j]: x[i, j].item() for j in range(self.n_qubits)}
            out.append(self.estimator_qnn(params))
        return torch.tensor(out, device=x.device).view(bsz, -1)

__all__ = ["QuantumFilter", "EstimatorQNNWrapper"]
