import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions and optional down‑sampling."""
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
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
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


class HybridFunction(torch.autograd.Function):
    """Differentiable dense head using a sigmoid activation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Linear head with optional batch‑norm and dropout."""
    def __init__(self, in_features: int, shift: float = 0.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.bn = nn.BatchNorm1d(1) if dropout > 0.0 else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        x = self.linear(logits)
        x = self.bn(x)
        x = self.dropout(x)
        return HybridFunction.apply(x, self.shift)


class QCNet(nn.Module):
    """
    Residual CNN followed by a hybrid dense head.
    The architecture mirrors the original seed but adds batch‑norm, dropout,
    and residual connections for improved expressivity.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.res1 = ResidualBlock(6, 6, stride=1)
        self.res2 = ResidualBlock(6, 6, stride=1)

        # The feature map size after conv/pool layers is 6×7×7 = 294
        self.fc1 = nn.Linear(294, 120, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.drop_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(84)
        self.drop_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(84, 1, bias=False)

        self.hybrid = Hybrid(1, shift=0.0, dropout=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)
        x = self.fc3(x)
        prob = self.hybrid(x)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["ResidualBlock", "HybridFunction", "Hybrid", "QCNet"]
