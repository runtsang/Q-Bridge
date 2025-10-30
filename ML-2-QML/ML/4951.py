import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class UnifiedQuantumHybridModel(nn.Module):
    """
    Classical hybrid model that fuses a CNN feature extractor, an LSTM encoder,
    and an optional RBF kernel for similarity analysis.
    """
    def __init__(self,
                 n_channels: int = 1,
                 n_classes: int = 4,
                 hidden_dim: int = 64,
                 seq_len: int = 1,
                 n_qubits: int = 0,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.seq_len = seq_len

        # CNN backbone identical to QuantumNAT
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU()
        )

        # LSTM encoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.norm = nn.BatchNorm1d(n_classes)

        # Kernel parameters
        self.kernel_gamma = kernel_gamma

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise RBF kernel between two batches."""
        x = x.unsqueeze(1)  # (N, 1, D)
        y = y.unsqueeze(0)  # (1, M, D)
        diff = x - y
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-self.kernel_gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (seq_len, batch, channels, H, W)
        Returns:
            logits: (batch, n_classes)
            kernel: (batch, batch) if n_qubits > 0 else None
            states: (h_n, c_n) from LSTM
        """
        seq_len, batch, _, _, _ = x.shape

        # Feature extraction per time step
        features = []
        for t in range(seq_len):
            feat = self.backbone(x[t])          # (batch, 16, 7, 7)
            feat = feat.view(batch, -1)         # (batch, 16*7*7)
            feat = self.feature_proj(feat)      # (batch, hidden_dim)
            features.append(feat.unsqueeze(0))  # (1, batch, hidden_dim)
        features = torch.cat(features, dim=0)    # (seq_len, batch, hidden_dim)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(features)  # h_n: (1, batch, hidden_dim)
        hidden = h_n.squeeze(0)                      # (batch, hidden_dim)

        # Classification
        logits = self.classifier(hidden)             # (batch, n_classes)
        logits = self.norm(logits)
        logits = F.log_softmax(logits, dim=1)

        # Kernel matrix if quantum path enabled
        if self.n_qubits > 0:
            kernel = self._rbf_kernel(hidden, hidden)  # (batch, batch)
        else:
            kernel = None

        return logits, kernel, (h_n, c_n)
