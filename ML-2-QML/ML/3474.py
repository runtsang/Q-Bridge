import torch
from torch import nn

class HybridConvEstimator(nn.Module):
    """
    Classical hybrid regressor that emulates a quantum‑inspired
    convolution‑estimator.  The module consists of a single 2‑D
    convolution that produces a scalar feature per sample, followed
    by a small MLP that mirrors the architecture of the
    EstimatorQNN example.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    conv_threshold : float, default 0.0
        Threshold applied inside the sigmoid activation to
        mimic the quantum thresholding behaviour.
    """
    def __init__(self, kernel_size: int = 2, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.conv_threshold = conv_threshold
        # Mimic the EstimatorQNN fully‑connected layers
        self.regressor = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        feat = self.conv(x)
        feat = torch.sigmoid(feat - self.conv_threshold)
        feat = feat.view(feat.size(0), -1)  # flatten to (batch, 1)
        return self.regressor(feat)

    def run(self, data: torch.Tensor) -> float:
        """
        Convenience wrapper that accepts a 2‑D array and returns a scalar.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            The scalar regression output.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return self.forward(tensor).item()

__all__ = ["HybridConvEstimator"]
