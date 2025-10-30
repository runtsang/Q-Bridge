import torch
import torch.nn as nn
import torch.nn.functional as F
from Conv import Conv

class HybridCNNQFCModel(nn.Module):
    """
    Classical hybrid model that integrates:
    * Two convolutional blocks (mirroring the original QFCModel).
    * A learnable quantum‑inspired filter implemented by Conv().
    * A fully connected head producing four outputs.
    The filter output is concatenated to the flattened features before the head.
    """

    def __init__(self, conv_filter_kernel: int = 2, conv_filter_threshold: float = 0.0):
        super().__init__()
        # Conv block 1
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Conv block 2
        self.features2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Quantum‑inspired filter (classical emulation)
        self.quantum_filter = Conv(kernel_size=conv_filter_kernel, threshold=conv_filter_threshold)

        # Fully connected head
        # Input size: 16 * 7 * 7 = 784 + 1 filter feature = 785
        self.fc = nn.Sequential(
            nn.Linear(785, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.features1(x)
        x = self.features2(x)

        # Global average pooling to feed into quantum filter
        pooled = F.avg_pool2d(x, kernel_size=6).view(x.size(0), -1)

        # Apply classical quantum filter
        filter_out = self.quantum_filter.run(pooled.cpu().numpy())
        filter_tensor = torch.tensor([filter_out] * x.size(0), device=x.device, dtype=x.dtype).unsqueeze(1)

        # Concatenate filter output
        out = torch.cat([x.view(x.size(0), -1), filter_tensor], dim=1)

        # Fully connected projection
        out = self.fc(out)
        return self.norm(out)

__all__ = ["HybridCNNQFCModel"]
