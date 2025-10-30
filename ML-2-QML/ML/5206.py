import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter with a learnable bias and threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # (batch, 1)

class SelfAttentionBlock(nn.Module):
    """Classical self‑attention that mirrors the quantum self‑attention circuit."""
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, embed_dim)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)  # (batch, seq_len, embed_dim)

class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))  # (batch, embed_dim)

class ClassicalQLSTM(nn.Module):
    """Drop‑in classical LSTM that mimics the interface of the quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return out  # (batch, seq_len, hidden_dim)

class HybridConvQLSTM(nn.Module):
    """Combined module that chains convolution, attention, QCNN, and LSTM."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 embed_dim: int = 8,
                 hidden_dim: int = 32,
                 n_qubits: int = 0):
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.embed = nn.Linear(1, embed_dim)  # embed conv output
        self.attn = SelfAttentionBlock(embed_dim)
        self.qcnn = QCNNModel(embed_dim)
        self.lstm = ClassicalQLSTM(embed_dim, hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input shape: (batch, 1, H, W)
        Returns: (batch, hidden_dim)
        """
        # Convolution
        conv_out = self.conv(x)  # (batch, 1)
        # Embedding
        embedded = self.embed(conv_out)  # (batch, embed_dim)
        # Add sequence dimension
        seq = embedded.unsqueeze(1)  # (batch, 1, embed_dim)
        # Self‑attention
        attn_out = self.attn(seq)  # (batch, 1, embed_dim)
        # Flatten for QCNN
        flattened = attn_out.view(attn_out.size(0), -1)  # (batch, embed_dim)
        # QCNN
        qcnn_out = self.qcnn(flattened)  # (batch, embed_dim)
        # Prepare for LSTM: need (batch, seq_len, input_dim)
        lstm_in = qcnn_out.unsqueeze(1)  # (batch, 1, embed_dim)
        # LSTM
        lstm_out = self.lstm(lstm_in)  # (batch, 1, hidden_dim)
        return lstm_out.squeeze(1)  # (batch, hidden_dim)

def Conv():
    """Factory returning the hybrid module."""
    return HybridConvQLSTM()
