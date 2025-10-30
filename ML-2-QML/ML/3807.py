import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQLSTM(nn.Module):
    """Hybrid LSTM with a classical convolution front‑end and a standard LSTM core.
    The architecture mirrors the quantum version but replaces all quantum sub‑modules
    with classical equivalents.  It is fully compatible with the original QLSTM
    interface and can be used as a drop‑in replacement or as a baseline for
    quantum experiments.
    """
    def __init__(self,
                 hidden_dim: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 conv_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 image_size: int = 28,
                 batch_first: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.n_qubits = n_qubits  # kept for API compatibility
        self.batch_first = batch_first

        # Classical analogue of the Quanvolution filter
        self.conv = nn.Conv2d(1, conv_channels,
                              kernel_size=conv_kernel,
                              stride=conv_stride)
        self.flatten = nn.Flatten()

        # Compute the flattened feature dimension
        conv_out = image_size // conv_stride
        feature_dim = conv_channels * conv_out * conv_out

        # Classical LSTM core
        self.lstm = nn.LSTM(feature_dim,
                            hidden_dim,
                            batch_first=batch_first)

        # Output head
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self,
                images: torch.Tensor,
                states: tuple | None = None) -> torch.Tensor:
        """
        Expect `images` of shape ``(batch, seq_len, 1, H, W)`` if ``batch_first=True``
        otherwise ``(seq_len, batch, 1, H, W)``.  The method flattens each frame,
        runs it through a convolutional front‑end, and feeds the resulting
        sequence to the LSTM core.  It returns log‑softmax logits over the tagset.
        """
        if self.batch_first:
            batch, seq_len, c, h, w = images.shape
            x = images.view(batch * seq_len, c, h, w)
        else:
            seq_len, batch, c, h, w = images.shape
            x = images.view(seq_len * batch, c, h, w)

        # Convolution + flatten
        x = self.conv(x)
        x = self.flatten(x)  # shape (batch*seq_len, feature_dim)

        # Restore sequence structure
        if self.batch_first:
            x = x.view(batch, seq_len, -1)
        else:
            x = x.view(seq_len, batch, -1)

        # LSTM core
        lstm_out, _ = self.lstm(x, states)

        # Tagging head
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

# Backwards compatibility aliases
QLSTM = HybridQLSTM
LSTMTagger = HybridQLSTM

__all__ = ["QLSTM", "LSTMTagger", "HybridQLSTM"]
