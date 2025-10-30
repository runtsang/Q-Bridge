import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Depth‑wise residual block used in the CNN backbone."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class DepthwiseSeparableCNN(nn.Module):
    """CNN backbone inspired by the Quantum‑NAT model but with depth‑wise separable convs."""
    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # depth‑wise conv
        self.dw_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3,
                                 stride=1, padding=1, groups=base_channels,
                                 bias=False)
        # point‑wise conv
        self.pw_conv = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1,
                                 bias=False)
        self.bn_pw = nn.BatchNorm2d(base_channels * 2)
        self.res_block = ResidualBlock(base_channels * 2, base_channels * 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dw_conv(out)
        out = self.pw_conv(out)
        out = self.bn_pw(out)
        out = self.relu(out)
        out = self.res_block(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

class ClassicalQuantumKernel(nn.Module):
    """Classical linear layer that mimics the quantum kernel output."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class QuantumHybridNet(nn.Module):
    """
    Hybrid architecture that merges ideas from the four seed references:
      * Classical CNN + FC (Quantum‑NAT) – the backbone is a depth‑wise separable 2‑D convolution with a residual block.
      * Quantum regression (QModel) – the quantum kernel is approximated by a linear layer.
      * Quanvolution – the patch‑wise extraction is omitted here but the quantum kernel can be applied to any feature vector.
      * Fraud‑detector – a clipping mechanism is applied to all linear layers.
    The model exposes two heads: a classification head for discrete labels and an optional regression head for continuous targets.
    """
    def __init__(self,
                 num_classes: int = 10,
                 regression: bool = False,
                 n_wires: int = 16,
                 device: str | torch.device | None = None) -> None:
        super().__init__()
        self.backbone = DepthwiseSeparableCNN(in_channels=1, base_channels=8)
        # After avgpool the backbone outputs a 16‑dim vector (8*2)
        self.quantum_kernel = ClassicalQuantumKernel(16, n_wires)
        self.classifier = nn.Linear(n_wires, num_classes)
        self.regression = regression
        if regression:
            self.regressor = nn.Linear(n_wires, 1)
        self._clip_parameters()
        self.to(device or torch.device('cpu'))

    def _clip_parameters(self, bound: float = 5.0) -> None:
        """Clip weights of all linear layers to keep them bounded."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.clamp_(min=-bound, max=bound)
                if m.bias is not None:
                    m.bias.data.clamp_(min=-bound, max=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - logits (classification) or
            - (logits, regression_output) if regression=True.
        """
        features = self.backbone(x)          # shape: (batch, 16)
        quantum_features = self.quantum_kernel(features)  # shape: (batch, n_wires)
        logits = self.classifier(quantum_features)
        if self.regression:
            regress = self.regressor(quantum_features).squeeze(-1)
            return logits, regress
        return logits

__all__ = ["QuantumHybridNet"]
