"""HybridQCNet: classical implementation mirroring the quantum architecture.

The module defines a convolutional filter, a QCNN-inspired fully‑connected network,
and a hybrid head that emulates the quantum expectation layer with a differentiable
sigmoid.  The class layout matches the quantum counterpart so that both can be
instantiated and benchmarked side‑by‑side.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """Classical emulation of a quanvolution filter.

    The filter applies a learnable 2‑D convolution to each channel and
    returns the mean activation after a sigmoid threshold.  The number
    of output kernels is configurable so that the feature vector matches
    the QCNN input size.
    """
    def __init__(self, kernel_size: int = 2, num_kernels: int = 8, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.threshold = threshold
        self.conv = nn.Conv2d(1, num_kernels, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)  # (batch, num_kernels, 1, 1)


class QCNNModel(nn.Module):
    """Fully‑connected network that emulates the QCNN layers.

    The architecture follows the original QCNN paper: feature map → conv → pool
    repeated three times, followed by a sigmoid output.  The network is fully
    differentiable and can be trained with standard back‑propagation.
    """
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        outputs, = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a sigmoid head."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridFunction.apply(logits, self.shift)


class HybridQCNet(nn.Module):
    """Complete classical network that mirrors the quantum hybrid architecture."""
    def __init__(self):
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size=2, num_kernels=8, threshold=0.0)
        self.reduce = nn.Linear(3 * 8, 8)  # 3 channels × 8 kernels
        self.qcnn = QCNNModel()
        self.hybrid = Hybrid(in_features=1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 3, H, W)
        batch, channels, _, _ = x.shape
        conv_feats = []
        for c in range(channels):
            channel = x[:, c:c + 1, :, :]  # (batch, 1, H, W)
            conv_feats.append(self.conv_filter(channel))
        conv_feats = torch.cat(conv_feats, dim=1)  # (batch, 24, 1, 1)
        conv_feats = conv_feats.view(batch, -1)   # (batch, 24)
        reduced = self.reduce(conv_feats)         # (batch, 8)
        qcnn_out = self.qcnn(reduced)             # (batch, 1)
        logits = self.hybrid(qcnn_out)            # (batch, 1)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return probs


__all__ = ["HybridQCNet"]
