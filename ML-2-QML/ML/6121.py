import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler network that emulates a QCNN‑style feature extractor
    and outputs a 2‑class probability distribution.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolution‑like linear layers
        self.feature = nn.Sequential(
            nn.Linear(2, 8), nn.Tanh(),   # conv1
            nn.Linear(8, 8), nn.Tanh(),   # conv2
            nn.Linear(8, 4), nn.Tanh(),   # pool
        )
        # Classification head
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a probability distribution over two classes.
        """
        h = self.feature(x)
        logits = self.head(h)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
