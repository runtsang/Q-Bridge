import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """Classical sigmoid head that emulates a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class QCNNHybridGen(nn.Module):
    """
    Hybrid QCNN that blends dense classical layers with a quantum-inspired head.
    The network can operate in a purely classical mode or use a hybrid head that
    mimics the quantum expectation value via a differentiable sigmoid.
    """
    def __init__(self,
                 num_features: int = 8,
                 hidden_dim: int = 16,
                 shift: float = 0.0,
                 use_hybrid_head: bool = False):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim - 4), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim - 4, hidden_dim - 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim - 4, hidden_dim - 8), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim - 8, hidden_dim - 8), nn.Tanh())
        self.head = nn.Linear(hidden_dim - 8, 1)
        self.use_hybrid_head = use_hybrid_head
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        if self.use_hybrid_head:
            return HybridFunction.apply(logits, self.shift)
        else:
            return torch.sigmoid(logits)

def QCNN() -> QCNNHybridGen:
    """Factory for the hybrid QCNN."""
    return QCNNHybridGen(use_hybrid_head=True)

__all__ = ["QCNNHybridGen", "QCNN"]
