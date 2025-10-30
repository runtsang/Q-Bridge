"""ConvGen105: A hybrid classical convolution module with advanced features.

This module extends the original Conv filter by:
- supporting multiple kernel sizes in a single forward pass,
- using a learnable threshold updated by gradient descent,
- sharing weights across channels,
- providing a hook for hybrid optimisation.
"""

import torch
from torch import nn
from typing import Iterable, Sequence

class ConvGen105(nn.Module):
    def __init__(
        self,
        kernel_sizes: Sequence[int] = (2, 3),
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        weight_shared: bool = True,
        init_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.weight_shared = weight_shared
        # Create convolution layers for each kernel size
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, bias=True)
            self.convs.append(conv)
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
        # Optional weight sharing: if True, all convs share the same weights
        if weight_shared:
            first = self.convs[0]
            for conv in self.convs[1:]:
                conv.weight = first.weight
                conv.bias = first.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutions and return a mean activation map per kernel."""
        activations = []
        for conv in self.convs:
            logits = conv(x)
            act = torch.sigmoid(logits - self.threshold)
            activations.append(act.mean(dim=(2, 3)))  # mean over spatial dims
        # Stack activations: shape (batch, num_kernels)
        out = torch.stack(activations, dim=1)
        return out

    def hybrid_hook(self, optimizer: torch.optim.Optimizer, loss_fn, data, labels):
        """Example hook that runs a hybrid optimisation step."""
        optimizer.zero_grad()
        preds = self.forward(data)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
