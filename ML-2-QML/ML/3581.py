import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQLSTM(nn.Module):
    """
    Classical hybrid sampler: LSTM encoder + feed‑forward sampler.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.sampler = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: Tensor of shape (batch, seq_len, input_dim)
        Returns: Tensor of shape (batch, 2) with class probabilities.
        """
        lstm_out, _ = self.lstm(seq)
        final_hidden = lstm_out[:, -1, :]
        probs = F.softmax(self.sampler(final_hidden), dim=-1)
        return probs

def SamplerQNN() -> SamplerQLSTM:
    """
    Helper to match the original anchor API.
    """
    return SamplerQLSTM()

__all__ = ["SamplerQLSTM", "SamplerQNN"]
