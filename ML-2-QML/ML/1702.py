import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical sampler neural network with optional dropout and deeper hidden layers.
    Provides a `forward` method returning a probability distribution and a `sample`
    method that draws samples via multinomial sampling.
    """
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        probs = self.forward(inputs)
        samples = torch.multinomial(probs, num_samples, replacement=True)
        return samples.squeeze(-1)

__all__ = ["HybridSamplerQNN"]
