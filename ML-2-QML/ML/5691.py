import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SamplerQNN(nn.Module):
    """
    A flexible classical sampler network that can be used as a stand‑in for a quantum sampler during
    hybrid training.  The architecture is parameterised by *in_features* and *out_features* so that
    it can be reused in a variety of settings.  Dropout and optional batch‑normalisation provide
    additional regularisation, and a ``sample`` helper exposes a categorical sampler.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 4,
        out_features: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        layers = [nn.Linear(in_features, hidden_features), activation]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_features))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability vector over the output classes."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log‑probability of the classes."""
        probs = self.forward(x)
        return torch.log(probs + 1e-12)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw ``num_samples`` from the categorical distribution defined by the network output.
        Returns a tensor of shape (batch, num_samples) containing integer indices.
        """
        probs = self.forward(x)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).permute(1, 0)

__all__ = ["SamplerQNN"]
