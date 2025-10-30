import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler network with residual connections and dropout.
    Mirrors the quantum SamplerQNN but with a deeper, more expressive
    architecture suitable for hybrid training pipelines.
    """
    def __init__(self) -> None:
        super().__init__()
        # Main feed‑forward body
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Residual block with ReLU activations
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        logits = self.fc3(h)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
