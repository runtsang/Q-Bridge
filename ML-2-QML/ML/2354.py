import torch
from torch import nn
import torch.nn.functional as F

class ConvQLSTM(nn.Module):
    """
    Classical hybrid Conv-QLSTM module.
    Combines a 2x2 convolution filter with a classical LSTM for sequence tagging.
    The convolution filter applies a sigmoid activation with a configurable threshold,
    mirroring the quantum filter's thresholding behaviour.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 hidden_dim: int = 128, vocab_size: int = 5000,
                 tagset_size: int = 10):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, seq_len, 1, kernel_size, kernel_size)
        Returns: log-probabilities over tags for each time step.
        """
        batch, seq_len, c, h, w = x.shape
        # Reshape to process each time step independently
        x_reshaped = x.view(batch * seq_len, c, h, w)
        conv_out = self.conv(x_reshaped)          # (batch*seq_len, 1, 1, 1)
        conv_out = conv_out.view(batch, seq_len, -1)  # (batch, seq_len, 1)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        lstm_out, _ = self.lstm(conv_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["ConvQLSTM"]
