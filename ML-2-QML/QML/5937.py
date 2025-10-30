import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QNN
import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """1‑qubit variational quantum layer that takes a single classical feature as a rotation
    angle and outputs the expectation value of the Y Pauli operator. The layer is fully
    differentiable and can be embedded inside a PyTorch model."""
    def __init__(self):
        super().__init__()
        input_param = Parameter("input")
        weight_param = Parameter("weight")
        qc = QuantumCircuit(1)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.qnn = QNN(
            circuit=qc,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)

__all__ = ["EstimatorQNN"]
