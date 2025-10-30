import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# Classical self‑attention (mimics the quantum interface)
# ------------------------------------------------------------------
class ClassicalSelfAttention:
    """Deterministic self‑attention that accepts rotation and entangle
    parameters as numpy arrays, mirroring the quantum signature."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ------------------------------------------------------------------
# Classical quanvolution filter
# ------------------------------------------------------------------
class ClassicalQuanvolutionFilter(nn.Module):
    """2×2 patch convolution that flattens the output."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

# ------------------------------------------------------------------
# Classical fully‑connected projection
# ------------------------------------------------------------------
class ClassicalQFCModel(nn.Module):
    """Linear projection to 4 features with batch‑norm."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc(x))

# ------------------------------------------------------------------
# Classical LSTM (drop‑in replacement)
# ------------------------------------------------------------------
class ClassicalQLSTM(nn.Module):
    """Standard LSTM that reshapes the 4‑dimensional features."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out)

# ------------------------------------------------------------------
# Hybrid network
# ------------------------------------------------------------------
class HybridQuanvolutionNet(nn.Module):
    """Classical baseline that mirrors the quantum architecture."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.qfc = ClassicalQFCModel()
        self.lstm = ClassicalQLSTM(input_dim=4, hidden_dim=8, n_qubits=4)
        self.attn = ClassicalSelfAttention(embed_dim=4)
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convolutional feature extraction
        features = self.qfilter(x)
        # 2. Projection to 4‑dimensional space
        qfc_out = self.qfc(features)
        # 3. Sequential modeling (treated as a batch‑first sequence)
        lstm_out, _ = self.lstm(qfc_out.unsqueeze(1))
        # 4. Self‑attention (deterministic, using fixed random params)
        rotation = np.random.randn(12)
        entangle = np.random.randn(3)
        attn_out = self.attn.run(rotation, entangle, lstm_out.squeeze(1).detach().cpu().numpy())
        # 5. Final classification
        logits = self.classifier(torch.from_numpy(attn_out).float().to(x.device))
        return F.log_softmax(logits, dim=-1)
