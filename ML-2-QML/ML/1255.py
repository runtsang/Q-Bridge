import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Extended classical sampler network.
    Architecture:
        2 -> 8 -> 4 -> 2 with Tanh activations and dropout.
    Provides softmax probabilities over 2 classes.
    """
    def __init__(self, dropout_prob: float = 0.3, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class probabilities.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
