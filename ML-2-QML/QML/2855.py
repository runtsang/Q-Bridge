import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable quantum expectation via central difference."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, estimator: EstimatorQNN, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.estimator = estimator
        input_np = inputs.detach().cpu().numpy()
        if input_np.ndim == 1:
            input_np = input_np.reshape(-1, 1)
        expectation = estimator.predict(input_np)
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        estimator = ctx.estimator
        input_np = inputs.detach().cpu().numpy()
        if input_np.ndim == 1:
            input_np = input_np.reshape(-1, 1)
        pos = input_np + shift
        neg = input_np - shift
        exp_pos = estimator.predict(pos)
        exp_neg = estimator.predict(neg)
        grad = (exp_pos - exp_neg) / (2 * shift)
        grad = torch.tensor(grad, dtype=torch.float32, device=inputs.device)
        return grad * grad_output, None, None

class QuantumHybridLayer(nn.Module):
    """Quantum layer that forwards activations through an EstimatorQNN."""
    def __init__(self, shift: float = np.pi / 2):
        super().__init__()
        theta = Parameter("theta")
        qc = qiskit.QuantumCircuit(1)
        qc.h(0)
        qc.rx(theta, 0)
        observable = SparsePauliOp.from_list([("Z", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[theta],
            weight_params=[],
            estimator=estimator,
        )
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.estimator_qnn, self.shift)

class HybridCNN_QNN(nn.Module):
    """CNN backbone followed by a quantum expectation head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = QuantumHybridLayer(shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridCNN_QNN", "QuantumHybridFunction", "QuantumHybridLayer"]
