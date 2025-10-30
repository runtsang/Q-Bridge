import numpy as np
import torch
from torch import nn
from typing import Sequence

# --------------------------------------------------------------------------- #
#  Classical fraud‑detection style dense block
# --------------------------------------------------------------------------- #
class FraudLayer(nn.Module):
    """
    2‑D linear layer with Tanh activation, optional clipping, and a
    constant scale/shift buffer.  Mirrors the style used in the fraud
    detection seed, but with deterministic weights for reproducibility.
    """
    def __init__(self, clip: bool = False):
        super().__init__()
        weight = torch.tensor([[0.5, -0.3], [0.2, 0.8]], dtype=torch.float32)
        bias = torch.tensor([0.1, -0.1], dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.linear(inputs))
        outputs = outputs * self.scale + self.shift
        return outputs


# --------------------------------------------------------------------------- #
#  Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """
    Simple dot‑product self‑attention using NumPy arrays.  The rotation
    and entangle parameters are only used to demonstrate the interface;
    in practice they can be learned or randomly sampled.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
#  RBF Kernel (classical)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """
    Radial basis function kernel implemented as a torch module to keep
    the API consistent with the quantum kernel class in the QML file.
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


# --------------------------------------------------------------------------- #
#  Hybrid Convolutional Module
# --------------------------------------------------------------------------- #
class HybridConv128(nn.Module):
    """
    Combines a 2‑D convolution, optional self‑attention, and optional
    fraud‑detection style dense layers.  The module is fully
    differentiable and can be dropped into a larger PyTorch graph.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 threshold: float = 0.0,
                 use_attention: bool = True,
                 use_fraud_layers: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.threshold = threshold
        self.attention = ClassicalSelfAttention(embed_dim=kernel_size * kernel_size) if use_attention else None
        self.fraud_layers = nn.Sequential(FraudLayer(), FraudLayer()) if use_fraud_layers else None

    def forward(self, x: torch.Tensor) -> float:
        # Convolution + sigmoid thresholding
        out = self.conv(x)
        out = torch.sigmoid(out - self.threshold)

        # Flatten spatial dimensions for attention
        out = out.view(out.size(0), -1)

        # Apply self‑attention (using dummy params for demo)
        if self.attention:
            rot = np.random.randn(out.size(1), out.size(1))
            ent = np.random.randn(out.size(1), out.size(1))
            out = self.attention.run(rot, ent, out.detach().cpu().numpy())
            out = torch.as_tensor(out, dtype=torch.float32)

        # Pass through fraud‑detection inspired layers
        if self.fraud_layers:
            out = self.fraud_layers(out)

        # Return a scalar summary
        return out.mean().item()

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      gamma: float = 1.0) -> np.ndarray:
        """
        Compute the Gram matrix between two lists of tensors using the
        classical RBF kernel.  Mirrors the API of the QML kernel module.
        """
        kernel = RBFKernel(gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridConv128", "RBFKernel"]
