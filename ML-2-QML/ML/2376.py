import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention:
    """Simple dot‑product self‑attention using NumPy tensors."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridQLSTMTagger(nn.Module):
    """Hybrid LSTMTagger that can operate with classical or quantum gates
    and uses a self‑attention mechanism.  The attention block is
    classical in this module, mirroring the quantum interface of the
    original seed.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # choose classical LSTM; the quantum flag is ignored here
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.attention = ClassicalSelfAttention(embed_dim=embedding_dim)

    def forward(self, sentence: torch.Tensor,
                rotation_params: np.ndarray | None = None,
                entangle_params: np.ndarray | None = None) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).float()
        # default parameters if not supplied
        if rotation_params is None:
            rotation_params = np.zeros(self.attention.embed_dim * 3)
        if entangle_params is None:
            entangle_params = np.zeros(self.attention.embed_dim)
        # apply attention over embeddings
        attn_out = torch.as_tensor(
            self.attention.run(rotation_params, entangle_params,
                               embeds.detach().cpu().numpy()),
            dtype=torch.float32,
            device=embeds.device
        )
        # concatenate attention output with embeddings
        combined = torch.cat([embeds, attn_out], dim=-1)
        # feed into LSTM
        lstm_out, _ = self.lstm(combined.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTMTagger"]
