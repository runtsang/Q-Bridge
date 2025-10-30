import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdvancedSamplerQNN(nn.Module):
    """
    Flexible, trainable sampler that outputs a probability distribution over two classes.
    Supports configurable hidden layers and differentiable Gumbel‑softmax sampling.
    """

    def __init__(self, hidden_dims: list[int] | None = None):
        """
        Parameters
        ----------
        hidden_dims : list[int], optional
            Sequence of hidden layer sizes. Defaults to a single hidden layer of size 4.
        """
        super().__init__()
        hidden_dims = hidden_dims or [4]
        layers = []
        input_dim = 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        layers.append(nn.Linear(input_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the softmax probability distribution over the two outputs.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable sampling using the Gumbel‑softmax trick.
        """
        logits = self.net(inputs)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def nll_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Negative log‑likelihood loss for training.
        """
        probs = self.forward(inputs)
        log_probs = torch.log(probs + 1e-10)
        return -torch.mean(torch.sum(log_probs * targets, dim=-1))
