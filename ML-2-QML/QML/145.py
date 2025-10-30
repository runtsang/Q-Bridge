import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class QuantumCircuit:
    """Parameterized variational circuit on n qubits."""
    def __init__(self, n_qubits: int, layers: int = 2, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.layers = layers
        self.device = device
        self.qnode = self._create_qnode()

    def _ansatz(self, params: torch.Tensor):
        """Hardware‑efficient ansatz using Ry rotations and CNOT chains."""
        for i in range(self.layers):
            for j in range(self.n_qubits):
                qml.RY(params[i, j], wires=j)
            for j in range(self.n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])

    def _create_qnode(self):
        @qml.qnode(qml.device(self.device, wires=self.n_qubits), interface="torch")
        def circuit(params: torch.Tensor):
            self._ansatz(params)
            return qml.expval(qml.PauliZ(0))
        return circuit

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        return self.qnode(params)

class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the variational quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        out = ctx.circuit(inputs)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = torch.zeros_like(inputs)
        # Parameter‑shift rule for each parameter
        for idx in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                shift_vec = torch.zeros_like(inputs)
                shift_vec[idx, j] = shift
                f_plus = ctx.circuit(inputs + shift_vec)
                f_minus = ctx.circuit(inputs - shift_vec)
                grad_inputs[idx, j] = (f_plus - f_minus) / 2
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that maps classical features to quantum parameters."""
    def __init__(self, n_qubits: int, layers: int, shift: float):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.shift = shift
        param_size = n_qubits * layers
        self.param_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, param_size)
        )
        self.circuit = QuantumCircuit(n_qubits, layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, 1)
        params = self.param_mlp(inputs)  # shape (batch, param_size)
        outputs = []
        for i in range(inputs.size(0)):
            param = params[i].view(self.layers, self.n_qubits)
            out = self.circuit(param)
            outputs.append(out)
        return torch.stack(outputs).squeeze(-1)

class HybridQuantumClassifier(nn.Module):
    """CNN backbone + variational quantum head."""
    def __init__(self,
                 n_qubits: int = 4,
                 layers: int = 2,
                 shift: float = np.pi / 2,
                 device: str = "default.qubit"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits, layers, shift)

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
        q_out = self.hybrid(x)
        return torch.cat((q_out, 1 - q_out), dim=-1)

__all__ = ["HybridQuantumClassifier", "QuantumCircuit", "HybridFunction", "Hybrid"]
