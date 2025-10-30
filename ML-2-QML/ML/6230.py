import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class HybridQuantumActivation(Function):
    """
    Differentiable activation that evaluates a parameterised quantum circuit
    to produce a scalar expectation value. The backward pass implements
    the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_circuit, shift: float) -> torch.Tensor:
        ctx.quantum_circuit = quantum_circuit
        ctx.shift = shift
        angles = inputs.view(-1).tolist()
        exp_values = quantum_circuit.run(np.array(angles))
        result = torch.tensor(exp_values, dtype=torch.float32).view(-1, 1)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        quantum_circuit = ctx.quantum_circuit
        grad_inputs = torch.zeros_like(inputs)
        for idx, val in enumerate(inputs.view(-1)):
            right = quantum_circuit.run(np.array([val.item() + shift]))
            left  = quantum_circuit.run(np.array([val.item() - shift]))
            grad = (right - left) / 2.0
            grad_inputs[idx] = grad
        return grad_inputs.view_as(inputs) * grad_output, None, None

class UnifiedHybridLayer(nn.Module):
    """
    Hybrid fully‑connected layer that can operate in either a classical
    mode (linear + tanh) or a quantum mode where the activation is
    computed by a parameterised quantum circuit.
    """
    def __init__(self, in_features: int, out_features: int,
                 use_quantum: bool = False,
                 quantum_circuit=None,
                 shift: float = np.pi / 2):
        super().__init__()
        self.use_quantum = use_quantum
        self.shift = shift
        if use_quantum:
            if quantum_circuit is None:
                raise ValueError("quantum_circuit must be provided when use_quantum=True")
            self.quantum_circuit = quantum_circuit
        else:
            self.linear = nn.Linear(in_features, out_features)
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            angles = x.view(x.size(0), -1)
            return HybridQuantumActivation.apply(angles, self.quantum_circuit, self.shift)
        else:
            return self.tanh(self.linear(x.view(x.size(0), -1)))

class HybridCNN(nn.Module):
    """
    Convolutional backbone followed by a UnifiedHybridLayer head.
    """
    def __init__(self, use_quantum: bool = False,
                 quantum_circuit=None,
                 shift: float = np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy = torch.zeros(1, 3, 32, 32)
        x = self.conv1(dummy)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        flat_size = x.shape[1]

        self.fc1 = nn.Linear(flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = UnifiedHybridLayer(1, 1, use_quantum=use_quantum,
                                       quantum_circuit=quantum_circuit,
                                       shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        out = self.head(x)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["HybridQuantumActivation", "UnifiedHybridLayer", "HybridCNN"]
