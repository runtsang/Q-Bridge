import torch
from torch import nn
import torch.nn.functional as F

class HybridEstimator(nn.Module):
    """
    Classical hybrid estimator that combines a shallow feed-forward network for static
    features with a classical LSTM for sequence tagging. It is fully compatible with
    the original EstimatorQNN but offers additional sequence handling.
    """

    def __init__(self, static_input_dim: int = 2, seq_input_dim: int = 2,
                 hidden_dim: int = 16, tagset_size: int = 10) -> None:
        super().__init__()
        # Feed‑forward backbone (static part)
        self.static_net = nn.Sequential(
            nn.Linear(static_input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
        # Sequence module
        self.lstm = nn.LSTM(seq_input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, static_inputs: torch.Tensor, seq_inputs: torch.Tensor):
        """
        Args:
            static_inputs: Tensor of shape (batch, static_input_dim)
            seq_inputs: Tensor of shape (batch, seq_len, seq_input_dim)
        Returns:
            static_out: Tensor of shape (batch, 1)
            tag_logits: Tensor of shape (batch, seq_len, tagset_size)
        """
        static_out = self.static_net(static_inputs)
        lstm_out, _ = self.lstm(seq_inputs)
        tag_logits = self.hidden2tag(lstm_out)
        return static_out, tag_logits

def EstimatorQNN():
    """
    Factory function retained for backward compatibility. Returns an instance of
    HybridEstimator with default parameters, mimicking the original EstimatorQNN
    signature.
    """
    return HybridEstimator()

__all__ = ["HybridEstimator", "EstimatorQNN"]
