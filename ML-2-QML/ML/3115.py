import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridConvNet(nn.Module):
    """
    Classical hybrid convolutional network that incorporates a
    lightweight quantum‑inspired filter as a drop‑in replacement
    for a true quanvolution.  The network consists of three
    convolutional blocks followed by a fully‑connected head
    producing four output features with batch‑norm.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 conv_filters: int = 8,
                 kernel_size: int = 3,
                 quantum_filter_kernel: int = 2,
                 threshold: float = 0.0,
                 device: torch.device | str | None = None):
        super().__init__()
        self.device = torch.device(device or "cpu")
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv_filters, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=kernel_size,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Classical quantum‑inspired filter
        self.quantum_filter = self._build_quantum_filter(quantum_filter_kernel,
                                                         threshold)
        # Fully‑connected head
        conv_out_dim = conv_filters * 4 * 3 * 3  # assuming 28×28 input
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + 1, 64),  # +1 from quantum filter scalar
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.batch_norm = nn.BatchNorm1d(num_classes)

    def _build_quantum_filter(self, kernel_size: int, threshold: float):
        """
        Return a callable that mimics a 2‑D quantum filter using a
        small 2‑D convolution followed by a sigmoid activation.
        """
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        return lambda x: torch.sigmoid(conv(x) - threshold).mean()

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
            Output logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        feat = self.features(x)
        # Classical quantum‑inspired scalar
        q_scalar = self.quantum_filter(feat[:, :1, :, :])  # use first channel
        # Flatten and concatenate
        flat = feat.view(bsz, -1)
        concat = torch.cat([flat, q_scalar], dim=1)
        out = self.fc(concat)
        return self.batch_norm(out)

__all__ = ["HybridConvNet"]
