import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Enhanced classical quanvolution filter with residual connections and learnable weight scaling."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.res_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.res_bn1 = nn.BatchNorm2d(out_channels)
        self.res_bn2 = nn.BatchNorm2d(out_channels)
        self.res_weight = nn.Parameter(torch.tensor(1.0))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.base_conv(x)
        res = self.res_conv1(x)
        res = self.res_bn1(res)
        res = self.relu(res)
        res = self.res_conv2(res)
        res = self.res_bn2(res)
        out = self.res_weight * res + main
        return out.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Deep residual quanvolution classifier with a learnable linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.log_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features) * self.log_scale
        logits = self.dropout(logits)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
