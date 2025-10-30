import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d

class QuantumNATEnhanced(nn.Module):
    """Hybrid CNN‑Transformer‑Quantum model with improved feature extraction and quantum readout."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        # --- Classical backbone -------------------------------------------------
        # Residual CNN blocks
        self.res_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Global transformer encoder for long‑range dependencies
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Projection to 4‑dim latent space
        self.classical_proj = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.classical_norm = BatchNorm1d(4)

        # --- Quantum branch ----------------------------------------------------
        # Parameter‑shaped embedding (4‑dim) will be fed into a variational circuit
        self.quantum_dim = 4
        self.quantum_circuit = nn.Sequential(
            nn.Linear(4, 4),  # identity mapping for compatibility
            nn.Softplus()
        )
        # Simple variational ansatz – a single layer of parameterized rotations
        self.var_ansatz = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh()
        )
        self.quantum_norm = BatchNorm1d(4)

        # Final fusion layer
        self.fusion = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical path
        features = self.res_cnn(x)
        flattened = features.view(bsz, -1)
        # Transformer expects sequence dimension first
        seq = flattened.unsqueeze(0)  # (1, B, C)
        transformed = self.transformer(seq).squeeze(0)
        class_out = self.classical_proj(transformed)
        class_out = self.classical_norm(class_out)

        # Quantum path – emulate a variational circuit
        q_emb = self.quantum_circuit(class_out)
        q_out = self.var_ansatz(q_emb)
        q_out = self.quantum_norm(q_out)

        # Fuse
        fused = torch.cat([class_out, q_out], dim=1)
        out = self.fusion(fused)
        return out

__all__ = ["QuantumNATEnhanced"]
