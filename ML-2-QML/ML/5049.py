from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamplerQNN"]

class SamplerQNN(nn.Module):
    """
    Classical hybrid sampler network that emulates the original SamplerQNN
    while adding a quanvolution filter and a quantum‑inspired LSTM head.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        tagset_size: int = 17,
        vocab_size: int = 1000,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        # 1. Classical sampler – small feed‑forward network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # 2. Quanvolution feature extractor (classical conv)
        self.quanv = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=2,
            stride=2,
        )

        # 3. Fully‑connected head to match hidden dimension
        self.fc = nn.Linear(4 * 14 * 14, hidden_dim)

        # 4. LSTM backbone (classical)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # 5. Output tag projection
        self.tag_head = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (batch, 1, 28, 28)

        Returns:
            log‑softmax logits of shape (batch, tagset_size)
        """
        # Sample step – produce a 2‑dim latent vector (unused in this demo)
        latent = self.sampler(x.view(x.size(0), -1)[:, :2])  # dummy usage

        # Quanvolution step – extract local quantum‑like patches
        features = self.quanv(x).view(x.size(0), -1)

        # Fully connected projection
        hidden = F.relu(self.fc(features))

        # LSTM expects (batch, seq_len, input_size); seq_len=1
        lstm_out, _ = self.lstm(hidden.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)

        logits = self.tag_head(lstm_out)
        return F.log_softmax(logits, dim=-1)
