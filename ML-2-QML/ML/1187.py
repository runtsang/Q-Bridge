import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class QuanvolutionNet(nn.Module):
    """
    Classical extension of the original quanvolution idea.
    The network stacks two residual‑style conv blocks followed by a
    small MLP head.  It uses 2×2 convolutions to mimic the
    quantum‑inspired filter but replaces the kernel with a learnable
    residual connection and expands the feature dimensionality.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        # First 2×2 convolution (stride 2) reduces 28×28 → 14×14
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, padding=0)
        # Second 2×2 convolution (stride 2) reduces 14×14 → 7×7
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2, padding=0)

        # Residual connection from conv2 to conv3
        self.residual = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)

        # MLP head
        self.fc1 = nn.Linear(8 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.conv1(x))
        # Conv block 2
        x = F.relu(self.conv2(x))
        # Residual addition
        x = F.relu(self.residual(x) + x)
        # Flatten
        x = x.view(x.size(0), -1)
        # MLP
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
