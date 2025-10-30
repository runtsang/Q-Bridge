import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class QuanvolutionFilter(nn.Module):
    """
    Classical 2×2 convolutional filter with optional trainable attention.
    The original seed used a fixed Conv2d; here the kernel is learnable
    and each channel can be weighted by a scalar attention factor.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 attention: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=True)
        self.attention = attention
        if attention:
            # per‑channel scalar attention weight
            self.attn_weights = Parameter(torch.ones(out_channels, 1, 1))
        else:
            self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Flattened feature vector of shape (B, out_channels * H' * W').
        """
        feat = self.conv(x)
        if self.attention:
            feat = feat * self.attn_weights
        return feat.view(feat.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """
    Flexible classifier head that can be a single linear layer or a
    multi‑layer perceptron.  The seed used a single linear layer; we
    generalise it to allow hidden layers and dropout, making the head
    more expressive while staying purely classical.
    """
    def __init__(self,
                 in_features: int,
                 num_classes: int = 10,
                 hidden_layers: list[int] | None = None,
                 dropout: float = 0.0):
        super().__init__()
        hidden_layers = hidden_layers or []
        layers = []
        last_dim = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)

class QuanvolutionModule(nn.Module):
    """
    End‑to‑end hybrid module that stacks the classical quanvolution
    filter with the flexible classifier.  The interface mirrors the
    original seed but exposes additional knobs: attention on the
    filter and hidden layers in the classifier.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 attention: bool = True,
                 hidden_layers: list[int] | None = None,
                 dropout: float = 0.0):
        super().__init__()
        self.filter = QuanvolutionFilter(in_channels=in_channels,
                                         out_channels=out_channels,
                                         attention=attention)
        # 28×28 input → 14×14 patches after 2×2 conv stride 2
        in_features = out_channels * 14 * 14
        self.classifier = QuanvolutionClassifier(in_features=in_features,
                                                 hidden_layers=hidden_layers,
                                                 dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        return self.classifier(feats)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuanvolutionModule"]
