import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNN(nn.Module):
    """
    An extended classical sampler neural network.
    Builds upon the original 2→4→2 architecture by adding configurable hidden layers,
    dropout, and batch‑normalization. Includes convenience methods for sampling
    and computing cross‑entropy loss.
    """
    def __init__(self, hidden_dims=(8, 8), dropout=0.1, device='cpu'):
        super().__init__()
        layers = []
        in_dim = 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class probabilities.
        """
        return F.softmax(self.net(x.to(self.device)), dim=-1)

    def sample(self, n_samples: int, seed: int | None = None) -> torch.Tensor:
        """
        Sample discrete actions from the learned distribution.
        """
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            probs = self.forward(torch.randn(n_samples, 2, device=self.device))
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross‑entropy loss between logits and one‑hot targets.
        """
        return F.cross_entropy(logits, targets)
