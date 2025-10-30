import numpy as np
import torch
from torch import nn

class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected network combining
    • a classic linear layer,
    • a self‑attention block,
    • an RBF kernel layer,
    • fraud‑detection style affine layers.
    All components are pure PyTorch and can be trained end‑to‑end.
    """
    def __init__(self,
                 input_dim: int = 1,
                 embed_dim: int = 4,
                 kernel_gamma: float = 1.0,
                 fraud_clip: bool = True):
        super().__init__()
        # 1. Fully connected
        self.fcl = nn.Linear(input_dim, 1)

        # 2. Self‑attention
        self.attn = self._SelfAttention(embed_dim)

        # 3. RBF kernel
        self.kernel = self._RBFKernel(kernel_gamma)

        # 4. Fraud‑detection style layers
        self.fraud_layers = nn.ModuleList()
        self.fraud_clip = fraud_clip
        self.final_linear = nn.Linear(2, 1)

    # ------------------------------------------------------------------
    # 2. Self‑attention helper
    # ------------------------------------------------------------------
    class _SelfAttention(nn.Module):
        def __init__(self, embed_dim: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.query = nn.Linear(embed_dim, embed_dim)
            self.key   = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = self.query(x)
            k = self.key(x)
            scores = torch.softmax(q @ k.t() / np.sqrt(self.embed_dim), dim=-1)
            return scores @ x

    # ------------------------------------------------------------------
    # 3. RBF kernel helper
    # ------------------------------------------------------------------
    class _RBFKernel(nn.Module):
        def __init__(self, gamma: float):
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    # ------------------------------------------------------------------
    # 4. Fraud‑detection style affine layers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _build_fraud_layer(self, params, clip: bool):
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias  = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale  = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift  = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(inp))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def add_fraud_layer(self, params, clip: bool = True):
        """Append a fraud‑detection style layer to the network."""
        self.fraud_layers.append(self._build_fraud_layer(params, clip))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                kernel_points: torch.Tensor | None = None) -> dict:
        # Fully‑connected output
        fcl_out = torch.tanh(self.fcl(x))

        # Self‑attention output
        attn_out = self.attn(x)

        # Kernel evaluation (optional)
        if kernel_points is not None:
            batch = []
            for xi in x:
                vals = [self.kernel(xi, y) for y in kernel_points]
                batch.append(torch.cat(vals))
            kernel_out = torch.stack(batch)
        else:
            kernel_out = None

        # Fraud stack
        fraud_out = x
        for layer in self.fraud_layers:
            fraud_out = layer(fraud_out)

        # Final scalar prediction
        final = self.final_linear(fraud_out)

        return {
            "fcl": fcl_out,
            "attention": attn_out,
            "kernel": kernel_out,
            "fraud": final
        }

__all__ = ["HybridFCL"]
