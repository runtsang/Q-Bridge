import torch
from torch import nn

class ConvGen(nn.Module):
    """
    Drop‑in replacement for Conv that scales to multi‑resolution and learnable kernels.
    Parameters
    ----------
    kernel_sizes : Iterable[int]
        List of kernel sizes per depth level (default [3,5]).
    in_channels : int
        Number of input channels (default 1).
    out_channels : int
        Number of output channels per level (default 1).
    depthwise : bool
        If True, use depth‑wise separable convs (default True).
    attention : bool
        If True, compute a channel‑wise attention map over the final feature map.
    threshold : float
        Activation threshold used in the sigmoid gating.
    """
    def __init__(self,
                 kernel_sizes=[3,5],
                 in_channels=1,
                 out_channels=1,
                 depthwise=True,
                 attention=True,
                 threshold=0.0):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.layers = nn.ModuleList()
        for k in kernel_sizes:
            if depthwise:
                conv = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=k,
                                 padding=k//2,
                                 padding_mode='circular')
                pw = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               padding=0)
                self.layers.append(nn.Sequential(conv, pw))
            else:
                conv = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=k,
                                 padding=k//2,
                                 padding_mode='circular')
                self.layers.append(conv)
        if attention:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels * len(kernel_sizes), out_channels * len(kernel_sizes), kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.attn = None
        self.threshold = threshold

    def forward(self, x):
        """
        Forward pass through the multi‑resolution conv layers.
        Args:
            x: Tensor of shape (batch, in_channels, H, W)
        Returns:
            Tensor of shape (batch, out_channels * len(kernel_sizes), H, W)
        """
        outputs = []
        for layer in self.layers:
            out = layer(x)
            out = torch.sigmoid(out - self.threshold)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)
        if self.attn is not None:
            out = out * self.attn(out)
        return out
