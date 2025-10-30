import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler network enhanced with dropout, additional hidden layers
    and utility functions for sampling and loss computation.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [8, 4]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities over the action space."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the output distribution.
        Returns a tensor of shape (batch, n_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples=n_samples, replacement=True)

    def negative_log_likelihood(self, probs: torch.Tensor,
                                target: torch.Tensor) -> torch.Tensor:
        """
        Compute NLL for training.
        """
        return F.nll_loss(torch.log(probs + 1e-12), target)

    @staticmethod
    def load_pretrained(path: str) -> "SamplerQNN":
        """
        Load a pretrained SamplerQNN from a checkpoint and set to eval mode.
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = SamplerQNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

__all__ = ["SamplerQNN"]
