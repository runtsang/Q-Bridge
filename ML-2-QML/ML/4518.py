import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSelfAttention:
    """Classical hybrid model that bundles self‑attention, LSTM, QCNN and QNN sub‑modules."""

    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        # LSTM: classical or quantum‑inspired
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.qcnn = self._build_qcnn()
        self.estimator = self._build_estimator()
        self.n_qubits = n_qubits

    def _build_qcnn(self):
        class QCNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
                self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
                self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
                self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
                self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
                self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
                self.head = nn.Linear(4, 1)
            def forward(self, x):
                x = self.feature_map(x)
                x = self.conv1(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = self.pool2(x)
                x = self.conv3(x)
                return torch.sigmoid(self.head(x))
        return QCNNModel()

    def _build_estimator(self):
        class EstimatorNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2,8), nn.Tanh(),
                    nn.Linear(8,4), nn.Tanh(),
                    nn.Linear(4,1))
            def forward(self, x):
                return self.net(x)
        return EstimatorNN()

    def self_attention(self, inputs: torch.Tensor, rotation_params: torch.Tensor, entangle_params: torch.Tensor):
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(0))
        tag_logits = self.hidden2tag(lstm_out.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)
