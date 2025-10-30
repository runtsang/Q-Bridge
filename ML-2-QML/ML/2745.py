import torch
import torch.nn as nn
import numpy as np

class UnifiedHybridLayer(nn.Module):
    """
    Classical dense head that emulates the behaviour of the quantum FCL.
    Parameters
    ----------
    in_features : int
        Number of input features.
    shift : float, optional
        Bias shift applied before the sigmoid activation.
    """
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        values = torch.tensor(thetas, dtype=torch.float32).view(-1, 1)
        logits = self.linear(values)
        probs = torch.sigmoid(logits + self.shift)
        return probs.detach().numpy()
