import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit

class HybridAttention(nn.Module):
    """Classical attention module that re‑weights the penultimate feature map
    before it is fed into the hybrid head.  The attention vector is
    learned via a small MLP and can be turned off during ablation."""
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_features),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)
        return x * w

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Extended hybrid head that can operate in three modes:
    * quantum: uses a variational circuit
    * hybrid: uses the classical sigmoid head
    * classical: returns raw logits for a standard BCE loss."""
    def __init__(self, in_features: int, shift: float = 0.0, mode: str = "hybrid"):
        super().__init__()
        self.mode = mode
        self.shift = shift
        self.linear = nn.Linear(in_features, 1)
        if self.mode == "quantum":
            # Placeholder for a quantum circuit; real implementation in QML module
            self.quantum = None
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "quantum":
            # In practice, this would call the quantum circuit
            raise NotImplementedError("Quantum mode requires QML integration.")
        elif self.mode == "hybrid":
            logits = self.linear(inputs)
            return HybridFunction.apply(logits, self.shift)
        else:  # classical
            return self.linear(inputs)

class QCNet(nn.Module):
    """CNN backbone with an attention‑augmented hybrid head."""
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
        self.attn = HybridAttention(84)
        self.hybrid = Hybrid(self.fc3.out_features, shift=np.pi/2, mode="hybrid")
    def forward(self, inputs: torch.Tensor):
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
        attn_out = self.attn(x)
        out = self.hybrid(attn_out)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["HybridAttention", "HybridFunction", "Hybrid", "QCNet"]
