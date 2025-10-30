import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention:
    """
    Classical self‑attention module that mimics the quantum self‑attention interface.
    Rotation and entangle parameters are expected to be 2‑D tensors of shape
    (embed_dim, embed_dim) and are applied as linear projections of the input.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention over the flattened feature vector.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query projection.
        entangle_params : np.ndarray
            Parameters for the key projection.
        inputs : np.ndarray
            Input feature vector of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted representation of the input.
        """
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class HybridNATModel(nn.Module):
    """
    Classical hybrid model that integrates convolutional feature extraction,
    self‑attention, and a regression head.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Self‑attention parameters (trainable)
        self.rotation_params = nn.Parameter(
            torch.randn(4, 4, dtype=torch.float32), requires_grad=True
        )
        self.entangle_params = nn.Parameter(
            torch.randn(4, 4, dtype=torch.float32), requires_grad=True
        )
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)
        self.attn = ClassicalSelfAttention(embed_dim=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalized regression output of shape (batch, 1).
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        # Apply self‑attention on the flattened features
        attn_out = self.attn.run(
            rotation_params=self.rotation_params.detach().cpu().numpy(),
            entangle_params=self.entangle_params.detach().cpu().numpy(),
            inputs=flattened.detach().cpu().numpy(),
        )
        attn_tensor = torch.as_tensor(attn_out, dtype=torch.float32, device=x.device)
        # Regression
        out = self.regressor(attn_tensor)
        return self.norm(out)


__all__ = ["HybridNATModel"]
