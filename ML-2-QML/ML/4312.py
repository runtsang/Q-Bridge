"""Unified hybrid layer combining CNN, classical FC, and quantum‑encoded components."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

# Import quantum components
try:
    from quantum_components import QuantumEncoder, QuantumLSTM
except ImportError:
    # Dummy placeholders if quantum components are not available
    class QuantumEncoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def run(self, thetas):
            return np.zeros_like(thetas)

    class QuantumLSTM(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, inputs, states=None):
            return inputs, states

class UnifiedQuantumHybridLayer(nn.Module):
    """
    A hybrid model that merges:
    - Classical CNN feature extractor (inspired by QuantumNAT).
    - Classical fully‑connected projection to a 4‑dimensional vector.
    - Quantum encoder mapping the 4‑dimensional vector onto a small quantum device.
    - Optional variational LSTM gate for sequential processing.
    - Final classifier.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        use_variational_lstm: bool = False,
        lstm_hidden_dim: int = 4,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        # 1. Classical CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 2. Classical fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # 3. Quantum encoder
        self.quantum_encoder = QuantumEncoder(n_qubits=n_qubits)
        # 4. LSTM (classical or quantum)
        if use_variational_lstm:
            self.lstm = QuantumLSTM(input_dim=4,
                                    hidden_dim=lstm_hidden_dim,
                                    n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_size=4,
                                hidden_size=lstm_hidden_dim,
                                num_layers=lstm_num_layers,
                                batch_first=False,
                                bidirectional=lstm_bidirectional)
        # 5. Final classifier
        self.out = nn.Linear(lstm_hidden_dim, 10)  # Example output size

    def forward(
        self,
        x: torch.Tensor,
        lstm_states: tuple | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28) or similar.
        lstm_states : tuple
            Optional initial hidden and cell states for the LSTM.
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, num_classes).
        """
        # Feature extraction
        feat = self.features(x)  # (batch, 16, 7, 7)
        flat = feat.view(feat.size(0), -1)  # (batch, 16*7*7)
        # Classical projection
        proj = self.fc(flat)  # (batch, 4)
        # Quantum encoding
        with torch.no_grad():
            proj_np = proj.detach().cpu().numpy()
            q_out_np = self.quantum_encoder.run(proj_np)  # (batch, 4)
        q_out = torch.from_numpy(q_out_np).to(proj.device).float()
        # LSTM processing
        seq = q_out.unsqueeze(0)  # (seq_len=1, batch, 4)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, lstm_states = self.lstm(seq, lstm_states)
        else:
            lstm_out, lstm_states = self.lstm(seq, lstm_states)
        # Take last output
        lstm_last = lstm_out[-1]  # (batch, hidden_dim)
        logits = self.out(lstm_last)
        return logits

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper that forwards to `forward`.
        """
        return self.forward(x)

__all__ = ["UnifiedQuantumHybridLayer"]
