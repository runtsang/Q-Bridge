import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Depthwise separable residual block."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch!= out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out += residual
        out = self.relu(out)
        return out


class QFCModelExtended(nn.Module):
    """Enhanced classical CNN with depthwise separable residual blocks."""
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 4,
        num_blocks: int = 3,
        feature_ch: int = 32,
    ):
        super().__init__()
        blocks = [ResidualBlock(in_channels, feature_ch) for _ in range(num_blocks)]
        self.features = nn.Sequential(*blocks)
        self.fc = nn.Sequential(
            nn.Linear(feature_ch * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["QFCModelExtended"]
