from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

# ------------------------------------------------------------------
# Classical Self‑Attention (mirrors the Qiskit implementation)
# ------------------------------------------------------------------
class ClassicalSelfAttention:
    """
    Compute a self‑attention matrix using classical matrix algebra.
    Parameters for rotation and entanglement are unused but kept for API
    compatibility with the quantum version.
    """
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# ------------------------------------------------------------------
# Classical QCNN (fully‑connected emulation of the quantum CNN)
# ------------------------------------------------------------------
class QCNNModel(nn.Module):
    """
    Linear stack that mimics the depth and non‑linearity of the QCNN
    defined in the quantum reference.  The architecture is kept
    identical so that the quantum and classical variants can be
    swapped without changing the surrounding code.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# ------------------------------------------------------------------
# Classical Fully‑Connected Layer (quantum‑style interface)
# ------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    """
    A tiny linear layer that exposes a ``run`` method, mirroring the
    quantum FCL interface.  It is used as the final projection
    before the tag‑score layer.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


# ------------------------------------------------------------------
# Hybrid Sequence Tagger (classical backbone)
# ------------------------------------------------------------------
class SharedClassName(nn.Module):
    """
    A classical sequence‑tagging model that stitches together:
    * an embedding layer
    * a classical self‑attention block
    * a QCNN‑style feature extractor
    * a standard LSTM backbone
    * a fully‑connected output head
    The design mirrors the quantum variant but uses only PyTorch
    primitives, making it fully trainable on CPU/GPU.
    """
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embed_dim: int = 8,
        hidden_dim: int = 32,
        n_qubits: int = 0,  # unused but kept for API compatibility
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = ClassicalSelfAttention(embed_dim)
        self.qcnn = QCNNModel()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = FullyConnectedLayer()
        self.tag_layer = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # 1. Embed
        embeds = self.embedding(sentence)  # (seq_len, embed_dim)

        # 2. Self‑attention (deterministic placeholder parameters)
        rotation_params = np.random.randn(embeds.shape[1] * 3)
        entangle_params = np.random.randn(embeds.shape[1] * 3)
        attn_out = self.self_attention.run(rotation_params, entangle_params, embeds.detach().cpu().numpy())
        attn_tensor = torch.as_tensor(attn_out, device=embeds.device)

        # 3. QCNN feature extraction
        # QCNN expects input of shape (batch, 8).  We process token‑wise.
        qcnn_feats = []
        for token in attn_tensor:
            feat = self.qcnn(token.unsqueeze(0))
            qcnn_feats.append(feat.squeeze(0))
        qcnn_tensor = torch.stack(qcnn_feats, dim=0).unsqueeze(0)  # (1, seq_len, 1)

        # 4. LSTM
        lstm_out, _ = self.lstm(qcnn_tensor)

        # 5. Tag projection
        logits = self.tag_layer(lstm_out.squeeze(0))
        return F.log_softmax(logits, dim=1)


__all__ = ["SharedClassName"]
