import torch
from torch import nn
import numpy as np

class ConvFilter(nn.Module):
    """
    Enhanced 2‑D convolutional filter with optional quantum‑derived feature fusion.
    The filter supports multiple input and output channels, an adaptive sigmoid
    threshold, and can fuse a quantum feature map produced by a separate
    'QuanvCircuit' instance.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        threshold: float = 0.0,
        fuse_quantum: bool = False,
        quantum_circuit: object | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.threshold = threshold
        self.fuse_quantum = fuse_quantum
        self.quantum_circuit = quantum_circuit

        if fuse_quantum and quantum_circuit is None:
            raise ValueError("Quantum circuit must be provided when fuse_quantum=True")

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=True
        )
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the convolution and optional quantum fusion.
        """
        x = x.to(self.device)
        conv_out = self.conv(x)

        if self.fuse_quantum:
            # Extract patches
            patches = nn.functional.unfold(
                x, kernel_size=self.kernel_size, stride=1
            )  # shape (B, C*ks*ks, L)
            B, _, L = patches.shape
            H_out = int((x.shape[2] - self.kernel_size) / 1 + 1)
            W_out = int((x.shape[3] - self.kernel_size) / 1 + 1)

            # Compute quantum feature for each patch
            quantum_feats = torch.empty((B, 1, H_out, W_out), device=self.device)
            for b in range(B):
                for idx in range(L):
                    patch = patches[b, :, idx].cpu().numpy()
                    # reshape to (C, H, W) and flatten channel dimension
                    patch_reshaped = patch.reshape(
                        self.in_channels, self.kernel_size, self.kernel_size
                    )
                    flat = patch_reshaped.reshape(-1)
                    q_val = self.quantum_circuit.run(flat)  # scalar float
                    h = idx // W_out
                    w = idx % W_out
                    quantum_feats[b, 0, h, w] = q_val

            conv_out = conv_out + quantum_feats

        # Apply adaptive sigmoid threshold
        out = torch.sigmoid(conv_out - self.threshold)
        return out

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method that applies the filter to a 2‑D array and returns the mean activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        tensor = tensor.view(1, self.in_channels, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

__all__ = ["ConvFilter"]
