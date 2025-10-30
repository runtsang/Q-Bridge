"""
Classical hybrid quanvolutional model capable of classification or regression.

The backbone is a lightweight 2×2 convolution that reduces a 28×28 image to a
feature map of shape (4, 14, 14).  A task‑specific linear head produces
log‑softmax logits for classification or raw scores for regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical quanvolutional backbone + task‑specific head.

    Parameters
    ----------
    task : str, default='classify'
        'classify' or'regress'.
    n_classes : int, default=10
        Number of classes for classification. Ignored for regression.
    """

    def __init__(self, task: str = "classify", n_classes: int = 10) -> None:
        super().__init__()
        self.task = task

        # 1→4 channels, 2×2 kernel, stride 2
        self.backbone = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.feature_dim = 4 * 14 * 14

        if self.task == "classify":
            self.head = nn.Linear(self.feature_dim, n_classes)
        else:
            self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(1)
        logits = self.head(features)
        if self.task == "classify":
            return F.log_softmax(logits, dim=-1)
        else:
            return logits.squeeze(-1)

    def set_task(self, task: str, n_classes: int = 10) -> None:
        """Switch the network to a different prediction task."""
        if task not in ("classify", "regress"):
            raise ValueError('task must be "classify" or "regress"')
        self.task = task
        if task == "classify":
            self.head = nn.Linear(self.feature_dim, n_classes)
        else:
            self.head = nn.Linear(self.feature_dim, 1)


__all__ = ["QuanvolutionHybrid"]
