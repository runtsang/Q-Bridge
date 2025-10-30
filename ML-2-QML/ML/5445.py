import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class ClassicalSelfAttention(nn.Module):
    """Trainable self‑attention module that mirrors the quantum version."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, self.rotation)
        key   = torch.matmul(inputs, self.entangle)
        scores = F.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

class FullyConnectedLayer(nn.Module):
    """Simple linear layer with a tanh non‑linearity."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class HybridQCNet(nn.Module):
    """Classical convolutional network followed by a quantum‑inspired head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Feature extraction
        self.fc1 = FullyConnectedLayer(55815, 120)
        self.attention = ClassicalSelfAttention(embed_dim=120)
        self.fc2 = FullyConnectedLayer(120, 84)
        self.fc3 = FullyConnectedLayer(84, 1)

        self.hybrid = HybridFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.attention(x)
        x = self.fc2(x)
        x = self.fc3(x)

        probs = self.hybrid(x, 0.0)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridFunction", "ClassicalSelfAttention", "FullyConnectedLayer", "HybridQCNet"]
