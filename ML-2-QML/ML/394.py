import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    An enriched classical sampler network.
    Combines a multi‑layer perceptron with batch‑normalisation,
    dropout and a temperature‑controlled softmax output.
    Provides a ``sample`` helper that draws discrete samples
    according to the learned probability distribution.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 8),
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 device: str | torch.device | None = None):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.net = nn.Sequential(*layers)
        self.temperature = temperature
        self.device = (torch.device(device) if device else torch.device('cpu'))
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution over the
        *input_dim* classes.
        """
        logits = self.net(x.to(self.device))
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw *n_samples* discrete samples from the output distribution.
        Returns a tensor of shape (n_samples, batch_size, input_dim).
        """
        probs = self.forward(x)
        cat = torch.distributions.Categorical(probs)
        return cat.sample((n_samples,))
