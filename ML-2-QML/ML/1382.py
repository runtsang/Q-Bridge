"""Classical hybrid quanvolution with depthwise separable conv and flexible head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class QuanvolutionHybrid(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, patch_size=2, stride=2,
                 hidden_dim=256, num_classes=10, task='classification'):
        super().__init__()
        self.task = task
        self.conv = DepthwiseSeparableConv(in_channels, out_channels,
                                           kernel_size=patch_size, stride=stride)
        self.feature_dim = (28 // patch_size) ** 2 * out_channels
        self.fc = nn.Linear(self.feature_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes if task == 'classification' else hidden_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.head(x)
        if self.task == 'classification':
            return F.log_softmax(logits, dim=-1)
        else:
            return logits

QuanvolutionFilter = QuanvolutionHybrid
QuanvolutionClassifier = QuanvolutionHybrid
