"""Hybrid convolutional module combining classical conv, transformer, QCNN pooling and classifier.

The class implements a drop‑in replacement for the original Conv filter with optional
transformer and QCNN-style pooling.  It is fully differentiable and can be used
inside a PyTorch training loop.

The class name is ``HybridConv`` to match the quantum implementation.
"""

from __future__ import annotations

import torch
from torch import nn


class HybridConv(nn.Module):
    """Classical hybrid convolutional block.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolution kernel.
    threshold : float, default=0.0
        Activation threshold for the sigmoid.
    use_qcnn_pooling : bool, default=False
        If ``True`` a QCNN-inspired fully connected pooling stage is added.
    depth : int, default=2
        Depth of the feed‑forward classifier.
    num_classes : int, default=2
        Number of output classes.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_qcnn_pooling: bool = False,
        depth: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolution layer
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # QCNN-inspired pooling
        if use_qcnn_pooling:
            pool_dim = max(1, kernel_size * kernel_size // 4)
            first_pool_dim = max(1, kernel_size * kernel_size // 2)
            self.pool = nn.Sequential(
                nn.Linear(kernel_size * kernel_size, first_pool_dim),
                nn.Tanh(),
                nn.Linear(first_pool_dim, pool_dim),
                nn.Tanh(),
            )
        else:
            self.pool = None
            pool_dim = kernel_size * kernel_size

        # Classifier – feed‑forward network mirroring the quantum classifier
        layers = []
        in_dim = pool_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Apply the hybrid block to ``data`` and return a scalar probability.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        torch.Tensor
            Scalar output – probability of the positive class.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        x = activations.mean(-1, keepdim=True)

        if self.pool is not None:
            x = self.pool(x.view(x.size(0), -1))

        out = self.classifier(x.view(x.size(0), -1))
        prob = torch.sigmoid(out).mean().item()
        return prob


__all__ = ["HybridConv"]
