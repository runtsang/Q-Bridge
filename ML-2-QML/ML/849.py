import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvGen118(nn.Module):
    """Multi‑scale, optionally depth‑wise separable convolutional filter with a learnable sigmoid threshold.

    The class can be used as a drop‑in replacement for the original Conv() factory.
    It processes the input image with several kernel sizes in parallel, averages the
    resulting activations, and applies a sigmoid with a learnable threshold.
    """

    def __init__(
        self,
        base_kernel_size: int = 2,
        multi_scale: list[int] | None = None,
        depthwise: bool = False,
        threshold: float = 0.0,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()
        if multi_scale is None:
            multi_scale = [1, 3, 5]
        self.base_kernel_size = base_kernel_size
        self.multi_scale = multi_scale
        self.depthwise = depthwise
        self.learning_rate = learning_rate

        # Create a list of convolutional layers for each scale
        self.convs = nn.ModuleList()
        for k in self.multi_scale:
            if self.depthwise:
                # depth‑wise separable: depth‑wise conv followed by point‑wise conv
                depth_conv = nn.Conv2d(1, 1, kernel_size=k, groups=1, bias=True)
                point_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
                self.convs.append(nn.Sequential(depth_conv, point_conv))
            else:
                self.convs.append(nn.Conv2d(1, 1, kernel_size=k, bias=True))

        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute the scalar activation for a single image patch.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            2‑D array with shape (kernel_size, kernel_size) or a batch of such patches.

        Returns
        -------
        torch.Tensor
            Scalar activation value.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape (batch, channels, height, width)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(1)
        activations = []
        for conv in self.convs:
            logits = conv(data)
            act = torch.sigmoid(logits - self.threshold)
            activations.append(act.mean(dim=[0, 2, 3]))
        # Average across scales
        return torch.stack(activations, dim=0).mean()

    def run(self, data):
        """Convenience wrapper that returns a Python float."""
        return self.forward(data).item()

    def fit(self, data, target, epochs: int = 1):
        """Simple SGD update of the learnable parameters.

        Parameters
        ----------
        data : array‑like
            Input patches.
        target : float
            Desired activation value (e.g. 0.5).
        epochs : int
            Number of optimisation steps.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = F.mse_loss(self.forward(data), torch.tensor(target, dtype=torch.float32))
            loss.backward()
            optimizer.step()

def Conv():
    """Factory that returns a ConvGen118 instance with default settings."""
    return ConvGen118()
