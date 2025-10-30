import torch
from torch import nn, Tensor
import numpy as np

# ---------------------------------------------------------------------------
# Classical helpers – adapted from the seed modules
# ---------------------------------------------------------------------------

class SelfAttentionLayer(nn.Module):
    """
    Classical self‑attention wrapper that emulates the behaviour of the
    original SelfAttention helper.  The layer learns rotation and
    entanglement parameters and applies them to the input sequence.
    """
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Parameters are learnable and will be updated during training
        self.rotation_params = nn.Parameter(torch.randn(embed_dim * 3))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))
        # The underlying functional implementation
        self.attn = SelfAttention()

    def forward(self, x: Tensor) -> Tensor:
        # x has shape (batch, features)
        if x.shape[1]!= self.embed_dim:
            raise ValueError(f"Input feature size {x.shape[1]} must match embed_dim {self.embed_dim}")
        # Convert to numpy for the legacy helper
        out_np = self.attn.run(
            rotation_params=self.rotation_params.detach().cpu().numpy(),
            entangle_params=self.entangle_params.detach().cpu().numpy(),
            inputs=x.detach().cpu().numpy()
        )
        return torch.tensor(out_np, device=x.device, dtype=x.dtype)


class QuantumKernelLayer(nn.Module):
    """
    Implements a learnable RBF‑style quantum kernel.
    The kernel is evaluated as a pairwise similarity between the input
    and a set of prototype vectors.  This layer replaces the
    QuantumKernel in the original seed and provides a dense feature map
    that can be fed into the hybrid head.
    """
    def __init__(self, num_prototypes: int, feature_dim: int, gamma: float = 1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, dim), prototypes: (num, dim)
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)   # (batch, num, dim)
        dist2 = torch.sum(diff ** 2, dim=2)                     # (batch, num)
        return torch.exp(-self.gamma * dist2)                   # (batch, num)


class HybridHead(nn.Module):
    """
    Differentiable hybrid head that mimics the quantum expectation.
    It is a single linear layer followed by a sigmoid shift.
    """
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


# ---------------------------------------------------------------------------
# Main QCNN‑style model
# ---------------------------------------------------------------------------

class QCNNGen525Model(nn.Module):
    """
    A hybrid QCNN that combines:
      • Classical convolution‑pooling blocks (Linear + Tanh),
      • A learnable self‑attention layer,
      • A quantum‑kernel similarity map,
      • A hybrid expectation head.
    The architecture mirrors the original QCNN seed but adds richer
    representational power from the self‑attention and kernel modules.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_prototypes: int = 8,
        shift: float = 0.0,
    ):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 8, hidden_dim // 8), nn.Tanh())

        self.self_attention = SelfAttentionLayer(embed_dim=hidden_dim // 8)
        self.kernel_layer = QuantumKernelLayer(
            num_prototypes=num_prototypes,
            feature_dim=hidden_dim // 8,
            gamma=1.0,
        )
        # The hybrid head receives concatenated attention + kernel features
        self.hybrid_head = HybridHead(
            in_features=(hidden_dim // 8) + num_prototypes,
            shift=shift,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass that processes the input through the feature map,
        convolution‑pooling blocks, self‑attention, kernel mapping, and
        finally the hybrid head.  The output is a probability in [0,1].
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        attn = self.self_attention(x)
        kernel = self.kernel_layer(x)

        concat = torch.cat([attn, kernel], dim=1)
        prob = self.hybrid_head(concat)
        return prob


def QCNNGen525() -> QCNNGen525Model:
    """
    Factory function returning a fully configured instance of
    :class:`QCNNGen525Model`.  Mirrors the API of the original QCNN factory.
    """
    return QCNNGen525Model(input_dim=8)


__all__ = ["QCNNGen525Model", "QCNNGen525"]
