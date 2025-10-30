import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation head."""
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

class ClassicalSelfAttention:
    """Classical self‑attention block with learnable rotation and entanglement parameters."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.mm(inputs, self.rotation_params)
        key = torch.mm(inputs, self.entangle_params)
        scores = torch.softmax(torch.mm(query, key.t()) / np.sqrt(self.embed_dim), dim=-1)
        return torch.mm(scores, inputs)

class ConvFilter(nn.Module):
    """Classical 2×2 convolutional filter that emulates a quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class QCNNHybrid(nn.Module):
    """Hybrid convolutional network that mirrors the quantum QCNN architecture."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.self_attention = ClassicalSelfAttention(embed_dim=4)
        self.conv_filter = ConvFilter()
        self.head = nn.Linear(1, 1)
        self.shift = 0.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Self‑attention
        x = self.self_attention.run(x)
        # Quanvolution filter applied to each sample
        batch_size = x.shape[0]
        filtered = []
        for i in range(batch_size):
            arr = x[i].cpu().numpy().reshape(2, 2)
            filtered.append(self.conv_filter.run(arr))
        filtered = torch.tensor(filtered, dtype=torch.float32, device=x.device).unsqueeze(1)
        logits = self.head(filtered)
        probs = HybridFunction.apply(logits, self.shift)
        return probs

__all__ = ["HybridFunction", "ClassicalSelfAttention", "ConvFilter", "QCNNHybrid"]
