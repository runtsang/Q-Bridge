import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridConvModel(nn.Module):
    """Hybrid classical convolutional neural network combining a 2×2
    convolutional filter (from Conv.py) and a fully‑connected head
    inspired by Quantum‑NAT.  The network can be used as a drop‑in
    replacement for either the Conv filter or the Quantum‑NAT model.
    """
    def __init__(self, kernel_size: int = 2, conv_threshold: float = 0.0,
                 num_classes: int = 4) -> None:
        super().__init__()
        # 2×2 classical convolution filter
        self.conv_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = conv_threshold
        # Main CNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, 1, H, W) with values in [0, 1].

        Returns
        -------
        torch.Tensor
            Normalised logits of shape (B, num_classes).
        """
        # Apply 2×2 filter and threshold
        filt = torch.sigmoid(self.conv_filter(x) - self.threshold)
        # Pad filt to match input size
        pad_left = (x.shape[-1] - filt.shape[-1]) // 2
        pad_right = x.shape[-1] - filt.shape[-1] - pad_left
        pad_top = (x.shape[-2] - filt.shape[-2]) // 2
        pad_bottom = x.shape[-2] - filt.shape[-2] - pad_top
        filt = F.pad(filt, (pad_left, pad_right, pad_top, pad_bottom))
        # Element‑wise multiply filter with input to modulate features
        x = x * filt
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.fc(flattened)
        return self.norm(logits)

__all__ = ["HybridConvModel"]
