import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNNEnhanced(nn.Module):
    """
    An expanded classical sampler network that mirrors the original
    SamplerQNN but adds hidden layers, dropout, batch normalization
    and a sampling interface.

    The network maps a 2‑dimensional input to a 2‑dimensional probability
    vector using a 3‑layer MLP.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (8, 8),
                 output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over the
        output dimension.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network output for each input instance.

        Args:
            x: Tensor of shape (batch, input_dim)
            n_samples: Number of samples per instance

        Returns:
            Tensor of shape (batch, n_samples, output_dim) containing
            one‑hot encoded samples.
        """
        probs = self.forward(x)
        probs_np = probs.detach().cpu().numpy()
        batch = probs_np.shape[0]
        output_dim = probs_np.shape[1]
        samples = np.zeros((batch, n_samples, output_dim), dtype=np.float32)
        for i in range(batch):
            for j in range(n_samples):
                choice = np.random.choice(output_dim, p=probs_np[i])
                samples[i, j, choice] = 1.0
        return torch.tensor(samples)

__all__ = ["SamplerQNNEnhanced"]
