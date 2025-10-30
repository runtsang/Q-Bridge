from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvSamplerQNN(nn.Module):
    """
    Classical hybrid convolution + sampler network.
    Combines a 2‑D convolution with a tiny MLP that predicts
    a data‑dependent threshold for the sigmoid activation.
    """

    def __init__(self, kernel_size: int = 2, threshold_range: tuple[float, float] = (0.0, 2.0)) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold_min, self.threshold_max = threshold_range

        # Convolutional filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Sampler network that maps two input values to a threshold
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the mean activation after thresholding.
        """
        # Predict threshold from the first two pixels
        flat = data.flatten()
        input_vals = flat[:2].unsqueeze(0)  # shape (1,2)
        thresh = self.sampler(input_vals).item()
        thresh = torch.lerp(torch.tensor(self.threshold_min),
                            torch.tensor(self.threshold_max),
                            torch.tensor(thresh))

        # Convolution
        tensor = data.unsqueeze(0).unsqueeze(0).float()  # shape (1,1,k,k)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - thresh)
        return activations.mean()

__all__ = ["ConvSamplerQNN"]
