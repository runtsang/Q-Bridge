import torch
from torch import nn
from torch.nn import functional as F

class QCNN(nn.Module):
    """
    Enhanced classical QCNN analogue.

    Architecture:
        - Feature extractor: 1‑D convolution + batch norm + ReLU.
        - 3 residual blocks (conv → BN → ReLU → conv → BN → add skip).
        - Dropout after each block to improve generalisation.
        - Global average pooling followed by a linear head.
    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    base_channels : int, default 16
        Number of channels in the first conv layer; doubled each block.
    num_blocks : int, default 3
        Number of residual blocks.
    dropout : float, default 0.2
        Dropout probability.
    """
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 16,
                 num_blocks: int = 3,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for _ in range(num_blocks):
            out_ch = in_ch * 2
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_ch)
            )
            self.blocks.append(block)
            in_ch = out_ch
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, channels, length) for 1‑D data.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) with sigmoid activation.
        """
        x = self.feature(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = F.relu(x)
            x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return torch.sigmoid(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.feature[0].in_channels}, " \
               f"base_channels={self.feature[0].out_channels}, " \
               f"num_blocks={len(self.blocks)}, dropout={self.dropout.p})"
