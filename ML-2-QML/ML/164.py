import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNExtended(nn.Module):
    """A robust classical sampler that maps a 2‑dimensional input vector to a categorical distribution over 2 classes.

    The architecture is deeper than the original seed, includes batch‑norm and dropout for regularisation,
    and exposes utilities for sampling, KL‑divergence loss and a flexible forward that accepts either a
    single sample or a batch.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    hidden_dims : Sequence[int], default (8, 16, 8)
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after every hidden layer.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (8, 16, 8),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h),
                           nn.BatchNorm1d(h),
                           nn.ReLU(),
                           nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability vector for each input sample.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 2) or (2,)
            The input features.

        Returns
        -------
        probs : torch.Tensor of shape (B, 2) or (2,)
            Softmax‑normalised probabilities.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw samples from the categorical distribution defined by the network output.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        n_samples : int
            Number of draws per input instance.

        Returns
        -------
        samples : torch.LongTensor
            Tensor of shape (B, n_samples) containing class indices.
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples=n_samples, replacement=True)

    def kl_loss(self, target: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """KL‑divergence loss between target distribution and network output.

        Parameters
        ----------
        target : torch.Tensor
            Ground‑truth probability distribution.
        probs : torch.Tensor
            Network‑produced distribution.

        Returns
        -------
        loss : torch.Tensor
            Mean KL‑divergence over the batch.
        """
        eps = 1e-12
        t = torch.clamp(target, min=eps)
        p = torch.clamp(probs, min=eps)
        return torch.mean(torch.sum(t * torch.log(t / p), dim=-1))

__all__ = ["SamplerQNNExtended"]
