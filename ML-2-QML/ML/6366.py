import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN__gen179(nn.Module):
    """
    Classical sampler network producing a temperature‑scaled softmax
    distribution.  The network mirrors the original 2‑to‑4‑to‑2
    architecture but the output is concatenated with a mock quantum
    distribution so that downstream tasks can read both views.
    """

    def __init__(self,
                 in_features: int = 2,
                 hidden_size: int = 4,
                 temperature: float = 1.0,
                 device: str = "cpu") -> None:
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, in_features)
        )
        self.device = device
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns a concatenated distribution of classical softmax and
        a mock quantum distribution.  The output shape is (batch, 4)
        where the first two entries are the softmax probabilities
        and the last two are the quantum‑mimicking probabilities.
        """
        logits = self.net(inputs)
        softmax = F.softmax(logits / self.temperature, dim=-1)

        # Mock quantum distribution: use a sigmoid on the first logit
        # to produce a probability for outcome 0; the complementary
        # probability is for outcome 1.
        q0 = torch.sigmoid(logits[:, 0])
        q1 = 1 - q0
        qdist = torch.stack([q0, q1], dim=-1)

        return torch.cat([softmax, qdist], dim=-1)

__all__ = ["SamplerQNN__gen179"]
