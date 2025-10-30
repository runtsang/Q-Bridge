import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Classical convolutional filter inspired by the original Quanvolution example.
    The filter can be pre‑trained and optionally frozen for transfer‑learning.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2, bias: bool = True,
                 pretrained_state: dict | None = None, freeze: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, bias=bias)
        if pretrained_state is not None:
            self.load_state_dict(pretrained_state, strict=False)
        self.freeze = freeze
        if self.freeze:
            for p in self.conv.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

    def set_pretrained(self, state_dict: dict) -> None:
        self.load_state_dict(state_dict, strict=False)

    def freeze_filter(self) -> None:
        self.freeze = True
        for p in self.conv.parameters():
            p.requires_grad = False

    def unfreeze_filter(self) -> None:
        self.freeze = False
        for p in self.conv.parameters():
            p.requires_grad = True

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid neural network using QuanvolutionFilter followed by a linear head.
    """
    def __init__(self, input_shape: tuple = (1, 28, 28), num_classes: int = 10,
                 filter_kwargs: dict | None = None, head_kwargs: dict | None = None):
        super().__init__()
        if filter_kwargs is None:
            filter_kwargs = {}
        self.qfilter = QuanvolutionFilter(**filter_kwargs)
        dummy = torch.zeros(1, *input_shape)
        n_features = self.qfilter(dummy).shape[1]
        if head_kwargs is None:
            head_kwargs = {}
        self.linear = nn.Linear(n_features, num_classes, **head_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
