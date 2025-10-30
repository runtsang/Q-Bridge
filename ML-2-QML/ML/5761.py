import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """Hybrid classical self‑attention with optional CNN encoder."""
    def __init__(self, embed_dim: int = 4, n_qubits: int = 4, use_cnn: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.use_cnn = use_cnn
        if use_cnn:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.flat_dim = 16 * 7 * 7
        else:
            self.encoder = nn.Identity()
            self.flat_dim = None
        self.norm = nn.BatchNorm1d(embed_dim)

    def _project(self, x: torch.Tensor, params: np.ndarray) -> torch.Tensor:
        """Linear projection of inputs using reshaped parameters."""
        w = torch.from_numpy(params.reshape(self.embed_dim, -1)).float()
        return torch.matmul(x, w.t())

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute self‑attention over the flattened features.
        Parameters:
            inputs: Tensor of shape (B, C, H, W) or (B, N, D)
            rotation_params, entangle_params: NumPy arrays of shape (embed_dim * D,)
        Returns:
            Tensor of shape (B, embed_dim, N) if use_cnn else (B, embed_dim, D)
        """
        if self.use_cnn:
            feats = self.encoder(inputs)
            flattened = feats.view(feats.size(0), -1)
            query = self._project(flattened, rotation_params)
            key = self._project(flattened, entangle_params)
            value = flattened
        else:
            query = self._project(inputs, rotation_params)
            key = self._project(inputs, entangle_params)
            value = inputs
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        out = torch.matmul(scores, value)
        return self.norm(out)
