import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionQLSTM(nn.Module):
    """
    Hybrid self‑attention + LSTM module.
    Classical embeddings and linear projections are followed by a quantum
    self‑attention block (if n_qubits > 0).  The sequence is then processed
    by either a classical LSTM or the quantum‑gate LSTM defined in
    :mod:`quantum_modules`.  The module returns log‑probabilities over a
    tag set and can be used as a drop‑in replacement for the original
    SelfAttention and QLSTM modules.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Linear projections for query/key/value
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Attention module (classical or quantum)
        if self.n_qubits > 0:
            from.quantum_modules import QuantumSelfAttention
            self.attention = QuantumSelfAttention(seq_len=embed_dim)
        else:
            self.attention = None

        # LSTM / QLSTM
        if self.n_qubits > 0:
            from.quantum_modules import QLSTM
            self.lstm = QLSTM(embed_dim, hidden_dim, self.n_qubits)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Output head
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.LongTensor
            Long tensor of shape (seq_len, batch_size).

        Returns
        -------
        torch.Tensor
            Log‑probabilities over tags: shape (seq_len, batch_size, tagset_size)
        """
        # Embedding
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)

        # Classical query/key/value
        q = self.proj_q(embeds)
        k = self.proj_k(embeds)
        v = self.proj_v(embeds)

        seq_len, batch, _ = q.size()

        # Self‑attention
        if self.attention is None:
            # Classical scaled dot‑product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, v)
        else:
            # Quantum‑enhanced attention
            # Prepare inputs for the quantum routine
            flat_q = q.reshape(seq_len * batch, -1).cpu().numpy()
            flat_k = k.reshape(seq_len * batch, -1).cpu().numpy()
            flat_v = v.reshape(seq_len * batch, -1).cpu().numpy()

            # Random rotation/entanglement parameters for illustration
            rotation_params = np.random.randn(self.n_qubits * 3)
            entangle_params = np.random.randn(self.n_qubits - 1)

            # Run the quantum circuit (returns a probability matrix)
            probs_np = self.attention.run(rotation_params,
                                          entangle_params,
                                          flat_q,
                                          shots=256)  # shape (seq_len*batch, seq_len)
            probs = torch.tensor(probs_np, dtype=torch.float32,
                                 device=q.device).reshape(seq_len, batch, seq_len)
            context = torch.matmul(probs, v)

        # LSTM / QLSTM
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(context)
        else:
            lstm_out, _ = self.lstm(context)

        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["SelfAttentionQLSTM"]
