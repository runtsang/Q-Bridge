import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Iterable

# Import quantum utilities from the separate QML module
# The QML module is expected to expose a quantum kernel function and a hybrid circuit class
# These imports keep the ML module free of quantum dependencies
from quantum_module import quantum_kernel, HybridCircuit


class LearnableKernel(nn.Module):
    """
    A hybrid kernel that blends a classical RBF kernel with a quantum kernel
    computed by the QML module. The blending weight ``alpha`` controls the
    contribution of each component.
    """
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def classical(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted sum of classical and quantum kernels.

        Parameters
        ----------
        x, y : torch.Tensor
            Tensors of shape (batch, features). The method is vectorised to
            compute all pairwise kernels between the two inputs.
        """
        # Classical RBF component
        classical = self.classical(x, y)

        # Quantum component – delegated to the QML module
        quantum = quantum_kernel(x, y)

        return self.alpha * classical + (1.0 - self.alpha) * quantum


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  alpha: float = 0.5) -> np.ndarray:
    """
    Build a Gram matrix between two collections of feature vectors
    using the hybrid LearnableKernel.
    """
    kernel = LearnableKernel(gamma, alpha)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class ClassicalAttention(nn.Module):
    """
    A lightweight multi‑head self‑attention block that operates on the
    flattened feature vector produced by the fully‑connected layers.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim). For our use‑case
            ``seq_len`` is 1, so the attention reduces to a simple
            linear transformation.
        """
        attn_output, _ = self.attn(x, x, x)
        return attn_output


class HybridHead(nn.Module):
    """
    Wrapper around the QML HybridCircuit. The circuit is created once
    during initialisation and reused for every forward call.
    """
    def __init__(self, circuit: HybridCircuit) -> None:
        super().__init__()
        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum circuit for each sample in the batch.
        """
        outputs = []
        for inp in inputs:
            out = self.circuit.run(inp.tolist())
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32)


class HybridKernelAttentionNet(nn.Module):
    """
    A binary classifier that combines:
    * classical convolutional feature extraction,
    * a learnable kernel for similarity estimation,
    * a self‑attention module, and
    * a quantum‑parameterised hybrid head.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Attention block
        self.attention = ClassicalAttention(embed_dim=84, num_heads=1)

        # Quantum hybrid head – instantiated from the QML module
        self.hybrid_head = HybridHead(HybridCircuit())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire network.
        """
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

        # Attention – reshape to (batch, seq_len=1, embed_dim)
        x = x.unsqueeze(1)
        x = self.attention(x).squeeze(1)

        x = self.fc3(x)

        # Quantum hybrid head
        x = self.hybrid_head(x)

        # Convert to probabilities
        prob = torch.sigmoid(x)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = [
    "LearnableKernel",
    "kernel_matrix",
    "ClassicalAttention",
    "HybridHead",
    "HybridKernelAttentionNet",
]
