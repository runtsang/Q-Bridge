import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumHybridFunction(torch.autograd.Function):
    """Quantum expectation head using EstimatorQNN."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, estimator: EstimatorQNN, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.estimator = estimator
        thetas = inputs.detach().cpu().numpy()
        expectations = np.array([estimator.run([theta])[0] for theta in thetas])
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        estimator = ctx.estimator
        grads = []
        for theta in inputs.detach().cpu().numpy():
            right = estimator.run([theta + shift])[0]
            left = estimator.run([theta - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class QuantumHybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a quantum estimator."""
    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        theta = Parameter("theta")
        qc = torch.quantum.QuantumCircuit(1)  # placeholder for actual qiskit circuit
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[("Z", 1)],
            input_params=[theta],
            weight_params=[],
            estimator=estimator,
        )
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs.squeeze(), self.estimator_qnn, self.shift)

class HybridBinaryClassifier(nn.Module):
    """Convolutional network followed by a quantum hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
        )
        self.hybrid_head = QuantumHybridLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        probs = self.hybrid_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
