import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical 2D convolution filter with bias and threshold gating."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, 1, H, W) or (H, W)
        Returns: Tensor of shape (batch, 1, H-k+1, W-k+1) after conv and sigmoid threshold
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class ConvLSTM(nn.Module):
    """
    Classical convolutional LSTM for sequence of images.
    Each time step: apply ConvFilter, flatten, feed to LSTM.
    """
    def __init__(self,
                 hidden_dim: int = 32,
                 num_layers: int = 1,
                 kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 batch_first: bool = True):
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size=kernel_size, threshold=conv_threshold)
        # Assume input images are 28x28
        conv_out_size = (28 - kernel_size + 1) ** 2
        self.lstm = nn.LSTM(input_size=conv_out_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.out_linear = nn.Linear(hidden_dim, 1)
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (seq_len, batch, 1, H, W) if batch_first=False
           or (batch, seq_len, 1, H, W) if batch_first=True
        Returns: Tensor of shape (seq_len, batch, 1) if batch_first=False
                 or (batch, seq_len, 1) if batch_first=True
        """
        if not self.batch_first:
            seq_len, batch, C, H, W = x.shape
        else:
            batch, seq_len, C, H, W = x.shape

        conv_seq = []
        for t in range(seq_len):
            if self.batch_first:
                img = x[:, t]
            else:
                img = x[t]
            conv_out = self.conv_filter(img)  # (batch, 1, H-k+1, W-k+1)
            conv_out = conv_out.view(batch, -1)  # flatten
            conv_seq.append(conv_out.unsqueeze(0))  # (1, batch, features)

        conv_seq = torch.cat(conv_seq, dim=0)  # (seq_len, batch, features)
        lstm_out, _ = self.lstm(conv_seq)
        out = self.out_linear(lstm_out)  # (seq_len, batch, 1)
        return out
