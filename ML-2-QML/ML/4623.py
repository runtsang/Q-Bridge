import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical emulation of a quantum convolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class UnifiedModel(nn.Module):
    """
    Hybrid classical model combining convolution, fully‑connected projection and
    a classical LSTM gate stack.  The quantum variant can be swapped in by
    passing ``use_quantum_lstm=True`` when constructing the object.
    """
    def __init__(self, use_quantum_lstm: bool = False, n_qubits: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

        self.use_quantum_lstm = use_quantum_lstm
        if not use_quantum_lstm:
            # Classical LSTM with hidden size 4 (same as output dim)
            self.lstm = nn.LSTM(4, 4, batch_first=True)
        else:
            raise RuntimeError("Quantum LSTM requires the QML module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        feat = self.features(x)
        flat = feat.view(batch, -1)
        out = self.fc(flat)
        out = self.norm(out)

        # Treat the 4‑dim feature as a sequence of length 1 for the LSTM
        seq = out.unsqueeze(1)  # (batch, seq_len=1, 4)
        lstm_out, _ = self.lstm(seq)
        return lstm_out.squeeze(1)

__all__ = ["ConvFilter", "UnifiedModel"]
