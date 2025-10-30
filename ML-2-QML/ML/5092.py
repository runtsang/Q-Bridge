from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler mirroring the quantum architecture.
    Combines a shallow feed‑forward sampler, a QCNN‑style feature extractor,
    a small regression head inspired by EstimatorQNN, and an optional
    classical LSTM for sequence handling.

    Parameters
    ----------
    use_qubit : bool, optional
        Flag to indicate whether a quantum backend is desired.  In
        the classical implementation this flag has no effect but is
        kept for API compatibility.
    hidden_dim : int, optional
        Hidden dimension for the embedded LSTM.
    """

    def __init__(self, use_qubit: bool = False, hidden_dim: int = 4) -> None:
        super().__init__()
        self.use_qubit = use_qubit

        # Classic sampler (mimics SamplerQNN)
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # QCNN‑style feature extractor (classical analogue)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Regression head (mimics EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

        # Classical LSTM (drop‑in for QLSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classical hybrid sampler.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1) representing a regression
            value that approximates the quantum sampler probability.
        """
        # 1. Sample distribution
        sampled = F.softmax(self.sampler(x), dim=-1)

        # 2. Pad to length 8 for feature extractor
        pad = torch.zeros(x.size(0), 6, device=x.device)
        fea = torch.cat([sampled, pad], dim=-1)

        # 3. Feature extraction
        y = self.feature_map(fea)
        y = self.conv1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.conv3(y)

        # 4. LSTM over a sequence of length 1
        y = y.unsqueeze(0)  # seq_len=1
        lstm_out, _ = self.lstm(y)
        lstm_out = lstm_out.squeeze(0)

        # 5. Regression head
        out = self.regressor(lstm_out)
        return out

__all__ = ["HybridSamplerQNN"]
