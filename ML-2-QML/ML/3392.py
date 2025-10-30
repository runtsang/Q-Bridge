import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class HybridEstimatorQLSTM(nn.Module):
    """Hybrid neural network that fuses a classical feed‑forward regressor with a
    sequence tagger using a standard LSTM.  The estimator accepts arbitrary
    feature vectors, while the tagger maps sequences of token indices to
    tag distributions.  The architecture remains fully differentiable
    and can be used as a drop‑in replacement for the original EstimatorQNN.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_sizes: Tuple[int,...] = (8, 4),
                 lstm_hidden_dim: int = 16,
                 vocab_size: int = 1000,
                 tagset_size: int = 10):
        super().__init__()
        # feed‑forward regressor
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

        # sequence tagger
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tagset_size)

    def forward(self,
                features: torch.Tensor,
                seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = {}
        # regression path
        out['regression'] = self.regressor(features).squeeze(-1)

        # tagging path
        if seq is not None:
            embeds = self.embedding(seq)  # (batch, seq_len, input_dim)
            lstm_out, _ = self.lstm(embeds)  # (batch, seq_len, hidden_dim)
            out['tags'] = F.log_softmax(self.hidden2tag(lstm_out), dim=-1)
        return out

def EstimatorQNN() -> HybridEstimatorQLSTM:
    """Compatibility wrapper that returns a hybrid estimator."""
    return HybridEstimatorQLSTM()

__all__ = ["HybridEstimatorQLSTM"]
