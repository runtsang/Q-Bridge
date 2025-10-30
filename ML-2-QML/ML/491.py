import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class ConvGen128(nn.Module):
    """Hybrid convolutional filter with 128×128 kernel.

    This class extends the original 2×2 Conv filter to a configurable
    128×128 kernel.  It supports multi‑channel inputs, a learnable
    weight matrix, and an optional quantum post‑processing step.
    The forward method returns a scalar activation that can be used
    directly in a loss function.
    """
    def __init__(self,
                 kernel_size: int = 128,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 threshold: float = 0.0,
                 use_quantum: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              bias=True)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.conv(x)
        out = torch.sigmoid(out - self.threshold)
        out = out.mean(dim=[1, 2, 3])
        return out

    def run(self, data):
        """Convenience wrapper that accepts a numpy array or a torch tensor."""
        if isinstance(data, np.ndarray):
            data = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(data).item()

    def hybrid_forward(self, data, quantum_module):
        """Hybrid forward that first applies the classical convolution
        and then feeds the result into a quantum circuit."""
        class_out = self.forward(data)
        class_np = class_out.detach().cpu().numpy()
        q_out = quantum_module.run(class_np)
        return torch.as_tensor(q_out, dtype=torch.float32)

    def train_step(self,
                   data,
                   target,
                   loss_fn: nn.Module = nn.MSELoss(),
                   optimizer: torch.optim.Optimizer | None = None) -> float:
        """Simple training step that computes loss and back‑propagates."""
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output = self.forward(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()

def ConvGen128Factory(**kwargs):
    """Factory that returns a ConvGen128 instance."""
    return ConvGen128(**kwargs)

__all__ = ["ConvGen128", "ConvGen128Factory"]
