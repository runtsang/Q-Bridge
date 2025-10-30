import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvQLSTM(nn.Module):
    """
    Classical hybrid module: patch extraction → (classical) convolution → LSTM.
    Parameters
    ----------
    kernel_size : int
        Size of the convolution filter (patch size).
    threshold : float
        Threshold used only in the quantum variant (kept for API compatibility).
    n_qubits : int
        If >0 the quantum filter is activated; otherwise a classical Conv2d is used.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the embedding (unused in the ML variant but kept for interface consistency).
    """
    def __init__(self, kernel_size=2, threshold=0.0, n_qubits=0, hidden_dim=128, vocab_size=30522):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = n_qubits

        # Classical convolution (acts on each patch)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # LSTM that processes the flattened patch sequence
        self.lstm = nn.LSTM(input_size=kernel_size*kernel_size,
                            hidden_size=hidden_dim,
                            batch_first=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).
        Returns
        -------
        torch.Tensor
            LSTM outputs of shape (batch, seq_len, hidden_dim).
        """
        batch, _, H, W = x.shape

        # Extract non‑overlapping patches using unfold
        patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # patches shape: (batch, 1, outH, outW, k, k)
        patches = patches.contiguous().view(batch, -1, self.kernel_size, self.kernel_size)

        # Apply classical convolution to each patch
        # We add a channel dimension, apply conv, then squeeze it back
        conv_out = self.conv(patches.unsqueeze(1)).squeeze(1)  # shape (batch, seq_len, 1)

        # Flatten to vector (optional, keeps compatibility with LSTM input size)
        conv_out = conv_out.view(batch, conv_out.size(1), -1)

        # Pass through LSTM
        out, _ = self.lstm(conv_out)
        return out

__all__ = ["ConvQLSTM"]
