import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Quanvolution__gen364(nn.Module):
    """
    Classical convolutional filter extended with dropout and weight‑norm regularization.
    The filter learns a 2x2 kernel and applies dropout to the feature maps.
    """
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2,
                 dropout_prob=0.3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride)
        self.conv = nn.utils.weight_norm(self.conv)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.zeros_(self.conv.bias)
        init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='linear')
        init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = F.relu(features)
        features = self.dropout(features)
        features = features.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen364"]
