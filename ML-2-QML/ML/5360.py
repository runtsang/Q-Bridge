import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedHybridLayer(nn.Module):
    """
    Classical counterpart of UnifiedHybridLayer.
    Implements a dense stack with optional conv‑like feature extraction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classical network.
        """
        return self.extractor(x)
