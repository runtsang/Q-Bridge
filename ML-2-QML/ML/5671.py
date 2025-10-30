import numpy as np
import torch
from torch import nn
from typing import Iterable

class HybridConvFC(nn.Module):
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 n_features: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.linear = nn.Linear(n_features + 1, 1)  # +1 for conv output

    def run(self, data: np.ndarray, thetas: Iterable[float]) -> float:
        """Return a scalar output.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) representing the
            image patch to be processed.
        thetas : Iterable[float]
            External parameters that are fed into the linear read‑out.

        Returns
        -------
        float
            The result of the hybrid classical layer.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        conv_out = activations.mean().item()

        feature = torch.as_tensor([conv_out] + list(thetas),
                                  dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(feature)).mean(dim=0)
        return expectation.detach().numpy().item()

__all__ = ["HybridConvFC"]
