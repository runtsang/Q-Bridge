import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Optional

# --------------------------------------------------------------------------- #
# 1. Classical fully‑connected layer (derived from FCL.py)
# --------------------------------------------------------------------------- #
class ClassicalFCL(nn.Module):
    """A lightweight fully‑connected layer that mimics the quantum FCL interface."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the expectation value of a tanh‑activated linear transform."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# --------------------------------------------------------------------------- #
# 2. Classical RBF kernel (derived from QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel that can be used as a feature map."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that keeps API compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                 gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for two sequences of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 3. Classical transformer block (derived from QTransformerTorch.py)
# --------------------------------------------------------------------------- #
class HybridTransformerBlock(nn.Module):
    """
    Classical multi‑head attention + feed‑forward block with a simple residual
    connection.  This block mirrors the API of the quantum version but
    performs all operations on the CPU.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 4. Unified hybrid model
# --------------------------------------------------------------------------- #
class UnifiedHybridLayer(nn.Module):
    """
    A single module that stitches together:
    1. A classical fully‑connected layer with a parameter vector.
    2. An RBF kernel feature map.
    3. A transformer‑style block that may use quantum modules.
    4. A final classification head.
    """
    def __init__(self,
                 n_features: int = 1,
                 gamma: float = 1.0,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 ffn_dim: int = 64,
                 num_blocks: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.fcl = ClassicalFCL(n_features)
        self.kernel = Kernel(gamma)
        self.transformer = nn.Sequential(
            *[HybridTransformerBlock(embed_dim, num_heads, ffn_dim,
                                     dropout=dropout) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, thetas: Iterable[float], x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 1) Run the quantum‑parameterized FCL, 2) map the
        feature vector through the RBF kernel, 3) feed the result to the
        transformer, 4) classify.
        """
        # Step 1: classical FCL
        fcl_out = self.fcl.run(thetas)          # shape: (1,)
        # Step 2: expand via kernel: we need a batch dimension; use the same vector for all items
        fcl_tensor = torch.tensor(fcl_out, dtype=torch.float32).view(1, -1)
        kernel_out = self.kernel(fcl_tensor, fcl_tensor).unsqueeze(0)  # shape (1,1)
        # Step 3: prepare sequence for transformer: replicate kernel_out across seq length
        seq_len = x.size(1)
        seq = kernel_out.repeat(seq_len, 1).unsqueeze(0)  # shape (1, seq_len, 1)
        # Step 4: transformer
        transformed = self.transformer(seq)
        # Step 5: classification
        pooled = transformed.mean(dim=1)  # shape (1, embed_dim)
        return self.classifier(pooled)
