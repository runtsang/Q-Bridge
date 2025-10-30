import torch
import torch.nn as nn
import torch.nn.functional as F
from.HybridSamplerQNN_qml import HybridSamplerQNN as QuantumHybridSamplerQNN

class HybridSamplerQNN(nn.Module):
    """Hybrid sampler combining classical encoding, a quantum sampler, and a regression head."""
    def __init__(self, num_features: int = 2, num_wires: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 4),
            nn.Tanh(),
            nn.Linear(4, num_wires),
        )
        self.quantum_sampler = QuantumHybridSamplerQNN(num_wires)
        self.regression_head = nn.Linear(2 ** num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode classical input to quantum parameters
        params = self.encoder(x)
        # Duplicate params for weights (simple strategy)
        weight_params = torch.cat([params, params], dim=-1)
        probs = self.quantum_sampler(params, weight_params)
        # Regression prediction
        return self.regression_head(probs).squeeze(-1)

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data: features in [-1,1] and target as sin(sum)+0.1*cos(2*sum)."""
    x = torch.rand(samples, num_features) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y
