import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SamplerConfig:
    """Configuration for the SamplerQNN module."""
    input_dim: int = 2
    hidden_dim: int = 4
    output_dim: int = 2
    dropout: float = 0.0
    use_broadcast: bool = False

    def __post_init__(self):
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError(f"Invalid input_dim: {self.input_dim}")

class SamplerQNN(nn.Module):
    """Hybrid classical‑quantum sampler with trainable variational parameters."""
    def __init__(self, config: SamplerConfig = SamplerConfig()) -> None:
        super().__init__()
        self.config = config
        # Classical network
        self.classical_net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        # Quantum parameters (just a placeholder; actual circuit handled in QML part)
        self.weight_params = nn.Parameter(torch.randn(config.hidden_dim))
        self.register_parameter("weight_params", self.weight_params)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, probabilities)."""
        logits = self.classical_net(inputs)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, weight_decay: float = 0.0) -> torch.Tensor:
        """Cross‑entropy loss with optional L2 weight decay on quantum parameters."""
        ce = F.cross_entropy(logits, targets)
        l2 = weight_decay * torch.sum(self.weight_params ** 2)
        return ce + l2
