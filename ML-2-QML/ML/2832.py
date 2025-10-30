"""Hybrid regressor with classical backbone and optional quantum placeholders.

The class can be instantiated in three modes:
  - classic: simple feed‑forward network.
  - attention_q: classical backbone + placeholder for quantum attention (no effect).
  - quantum_ffn: classical backbone + placeholder for quantum feed‑forward block.

The quantum placeholders are no‑ops in the classical implementation, but the same class name is used in the QML module where the placeholders are replaced by actual quantum submodules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedRegressorQNN(nn.Module):
    """Simple regressor with optional quantum placeholders."""
    def __init__(self,
                 mode: str = "classic",
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 1,
                 n_heads: int = 2,
                 n_qubits: int = 4):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_qubits = n_qubits

        # Classical backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Output layer
        if mode == "classic":
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        elif mode == "attention_q":
            # placeholder: same classical output
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        elif mode == "quantum_ffn":
            # placeholder: same classical output
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical backbone and output layer."""
        x = self.backbone(x)
        return self.output_layer(x)

__all__ = ["UnifiedRegressorQNN"]
