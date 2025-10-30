import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer, QuantumCircuit, assemble, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator


def build_ansatz(n_qubits: int = 2):
    """Two‑qubit variational ansatz with entanglement."""
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    qc.cx(0, 1)
    theta_in = Parameter("theta_in")
    theta_w = Parameter("theta_w")
    qc.ry(theta_in, 0)
    qc.ry(theta_w, 1)
    qc.cx(0, 1)
    return qc, [theta_in, theta_w]


def create_estimator_qnn() -> EstimatorQNN:
    """Wrap the ansatz in Qiskit’s EstimatorQNN."""
    qc, params = build_ansatz()
    input_params = [params[0]]          # data input
    weight_params = [params[1]]         # trainable weight
    observable = Pauli("Y")             # Pauli‑Y expectation
    estimator = QiskitEstimator(backend=Aer.get_backend("aer_simulator"))
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards a scalar through EstimatorQNN."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, estimator_qnn: EstimatorQNN) -> torch.Tensor:
        ctx.estimator_qnn = estimator_qnn
        # EstimatorQNN expects a 2‑D array (batch, input_dim)
        preds = estimator_qnn.predict(inputs.detach().cpu().numpy().reshape(-1, 1))
        out = torch.tensor(preds, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.pi / 2
        grads = []
        for val in inputs.cpu().numpy().reshape(-1):
            plus = ctx.estimator_qnn.predict(np.array([[val + shift]]))
            minus = ctx.estimator_qnn.predict(np.array([[val - shift]]))
            grads.append(plus - minus)
        grad_input = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_input * grad_output, None


class Hybrid(nn.Module):
    """Quantum hybrid layer built on EstimatorQNN."""
    def __init__(self, estimator_qnn: EstimatorQNN) -> None:
        super().__init__()
        self.estimator_qnn = estimator_qnn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.estimator_qnn)


class ResidualBlock(nn.Module):
    """Residual block mirroring the classical counterpart."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x))) + x


class QCNet(nn.Module):
    """
    Symmetric CNN backbone feeding a quantum expectation head.
    The architecture is intentionally parallel to the classical version
    to ease direct comparison.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.resblock = ResidualBlock(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.estimator_qnn = create_estimator_qnn()
        self.hybrid = Hybrid(self.estimator_qnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = self.hybrid(x)
        return torch.cat((out, 1 - out), dim=-1)


__all__ = ["HybridFunction", "Hybrid", "ResidualBlock", "QCNet"]
