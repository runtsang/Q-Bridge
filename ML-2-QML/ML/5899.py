import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = self.generate_superposition_data(num_features, samples)

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class UnifiedRegressionLSTM(nn.Module):
    'Classical hybrid regression model that optionally embeds input tokens and processes them through an LSTM before a small regression head.'

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int | None = None, vocab_size: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        if embedding_dim is not None and vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = None

        lstm_input_dim = embedding_dim if embedding_dim is not None else input_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        'Forward pass: x is (batch, seq_len, input_dim) or (batch, seq_len) if embedding is used.'
        if self.embedding is not None:
            x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        out = self.regressor(lstm_out)  # (batch, seq_len, 1)
        return out.squeeze(-1)
