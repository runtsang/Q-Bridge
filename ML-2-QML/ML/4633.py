import torch
import numpy as np
from torch import nn

# Classical Conv filter (drop‑in replacement for a quanvolution)
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# RBF kernel (classical implementation that mimics the quantum style)
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Self‑attention module that ties convolution and kernel together
class SelfAttentionHybrid:
    """
    Classical self‑attention that uses a convolutional filter to extract
    token features, a kernel to compute similarity, and a linear map
    to generate queries/keys/values.
    """
    def __init__(self, embed_dim: int = 4, kernel_gamma: float = 1.0) -> None:
        self.embed_dim = embed_dim
        self.conv = ConvFilter()
        self.kernel = Kernel(gamma=kernel_gamma)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Not used directly in the classical branch but kept for API
            compatibility; can be applied to inputs if desired.
        entangle_params : np.ndarray
            First element scales the kernel gamma; remaining elements are
            ignored to maintain consistency with the quantum interface.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim). Each token is processed
            by the convolutional filter to produce a scalar feature.
        Returns
        -------
        np.ndarray
            Output of the self‑attention layer.
        """
        batch, seq_len, _ = inputs.shape
        # Rotate inputs if rotation parameters are provided (optional linear map)
        if rotation_params.size > 0:
            rot = torch.tensor(
                rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
            )
            inputs = torch.as_tensor(inputs) @ rot
        else:
            inputs = torch.as_tensor(inputs)

        # Convolutional feature extraction per token
        conv_feats = torch.zeros(batch, seq_len, dtype=torch.float32)
        for b in range(batch):
            for s in range(seq_len):
                token = inputs[b, s]
                conv_feats[b, s] = self.conv.run(token.numpy())

        # Kernel similarity matrix (batch‑wise)
        gamma = entangle_params[0] if entangle_params.size > 0 else self.kernel.ansatz.gamma
        flat_feats = conv_feats.reshape(-1, 1)
        K = kernel_matrix(flat_feats, flat_feats, gamma=gamma)
        K = torch.tensor(K, dtype=torch.float32).reshape(batch, seq_len, seq_len)

        # Self‑attention
        qkv = self.qkv(inputs)  # shape (batch, seq_len, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        scores = torch.softmax((q @ k.transpose(-2, -1)) * K, dim=-1)
        out = torch.matmul(scores, v)
        return out.numpy()
