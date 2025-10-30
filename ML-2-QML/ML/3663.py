import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler network with optional weight clipping, dropout, and
    per‑output scale/shift buffers.  The design follows the original
    SamplerQNN while incorporating concepts from the fraud‑detection
    analogue: bounded parameters, non‑linear activation, and a learned
    affine transform applied before the softmax.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 4,
                 output_dim: int = 2,
                 clip_value: float = 5.0,
                 dropout_prob: float = 0.1,
                 seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

        # Per‑output affine transformation (analogous to displacement in the
        # photonic fraud‑detection circuit).
        self.scale = nn.Parameter(torch.ones(output_dim), requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

        self.clip_value = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clip all weights and biases to keep parameters bounded.
        for name, param in self.net.named_parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)

        logits = self.net(x)
        logits = logits * self.scale + self.shift
        return F.softmax(logits, dim=-1)

    def reset_parameters(self) -> None:
        """
        Re‑initialise all linear layers and the affine buffers with bounded
        values.  This mirrors the fraud‑detection model's clipping behavior.
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.shift)

__all__ = ["SamplerQNN"]
