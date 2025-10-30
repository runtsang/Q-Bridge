import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNN(nn.Module):
    """
    Classical sampler network.
    Supports arbitrary hidden depths, residual connections, batch‑norm and
    dropout to improve generalisation.  The forward method returns a
    probability vector via softmax, and a ``sample`` helper draws discrete
    outcomes according to that distribution.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] | None = None,
                 output_dim: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (4,)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        # Residual connection from input to the last hidden layer
        self.res = nn.Linear(input_dim, prev_dim)
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h + self.res(x)
        out = self.out(h)
        return F.softmax(out, dim=-1)

    def sample(self,
               x: torch.Tensor,
               n_samples: int = 1,
               seed: int | None = None) -> np.ndarray:
        """
        Sample discrete outcomes from the softmax distribution produced by
        the network.  Useful for generating synthetic data or performing
        importance‑sampling style experiments.
        """
        if seed is not None:
            torch.manual_seed(seed)
        probs = self.forward(x).detach().cpu().numpy()
        return np.random.choice(len(probs[0]),
                                size=(x.shape[0], n_samples),
                                p=probs[0])

__all__ = ["SamplerQNN"]
