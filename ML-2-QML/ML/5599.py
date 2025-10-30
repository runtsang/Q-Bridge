import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter that emulates a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class SelfAttentionBlock(nn.Module):
    """Classical self‑attention block mirroring the quantum interface."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim),
                           dim=-1)
        return scores @ value

class QLSTMGen507(nn.Module):
    """Hybrid LSTM layer that optionally inserts a quantum‑style gate implementation."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 conv_kernel: int = 2,
                 attention_dim: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttentionBlock(attention_dim)
        self.n_qubits = n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        Returns attended hidden states of shape (batch, seq_len, hidden_dim).
        """
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out

class LSTMTaggerRegressor(nn.Module):
    """End‑to‑end sequence model that uses QLSTMGen507 as the core."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 task: str = "tagging") -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.core = QLSTMGen507(embedding_dim, hidden_dim,
                                n_qubits=n_qubits)
        self.task = task
        if task == "tagging":
            self.head = nn.Linear(hidden_dim, tagset_size)
        elif task == "regression":
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.emb(sentence)
        hidden = self.core(embeds)
        logits = self.head(hidden)
        if self.task == "tagging":
            return F.log_softmax(logits, dim=-1)
        else:
            return logits.squeeze(-1)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic dataset generator that mimics the quantum superposition
    distribution used in the QML reference.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for regression experiments."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = ["QLSTMGen507"]
