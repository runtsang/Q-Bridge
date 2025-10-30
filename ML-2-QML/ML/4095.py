"""Hybrid self‑attention module with classical back‑end.

The class :class:`SelfAttentionHybrid` implements a self‑attention
block, a classical LSTM and a linear classifier.  The public API
mirrors the quantum version so that downstream code can switch
between back‑ends without modification.

Typical usage
-------------
>>> import numpy as np
>>> from SelfAttention__gen062 import SelfAttentionHybrid
>>> model = SelfAttentionHybrid(embed_dim=4, hidden_dim=8, n_qubits=4)
>>> rot = np.random.randn(4*3)
>>> ent = np.random.randn(4-1)
>>> inputs = np.random.randn(2, 5, 4)  # batch, seq_len, embed_dim
>>> logits = model.run(rot, ent, inputs)
>>> logits.shape
(2, 5, 4)
"""

import torch
import torch.nn as nn
import numpy as np

class ClassicalSelfAttention:
    """Fast, NumPy‑based self‑attention that mimics the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Compute a self‑attention weighted sum.

        Parameters
        ----------
        rotation_params, entangle_params
            Accepted for API compatibility but ignored in the classical
            implementation.
        inputs
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        """
        # Use the rotation/entangle params to build query/key matrices
        # (this mimics the quantum rotation angles).
        query = torch.from_numpy(
            inputs @ rotation_params.reshape(self.embed_dim, -1)
        ).float()
        key = torch.from_numpy(
            inputs @ entangle_params.reshape(self.embed_dim, -1)
        ).float()
        value = torch.from_numpy(inputs).float()
        scores = torch.softmax(
            query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1
        )
        return (scores @ value).numpy()


class ClassicalQLSTM(nn.Module):
    """Classical LSTM that mimics the quantum LSTM interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class SelfAttentionHybrid:
    """Hybrid self‑attention module that can be instantiated in a
    classical or quantum back‑end."""
    def __init__(self, embed_dim: int, hidden_dim: int, n_qubits: int):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.attention = ClassicalSelfAttention(embed_dim)
        self.lstm = ClassicalQLSTM(embed_dim, hidden_dim, n_qubits)
        self.fc = nn.Linear(hidden_dim, 4)  # 4‑class output like Quantum‑NAT

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Run the classical pipeline.

        Parameters
        ----------
        rotation_params, entangle_params
            Parameters for the self‑attention block – passed through
            unchanged for API compatibility.
        inputs
            NumPy array of shape ``(batch, seq_len, embed_dim)``.
        """
        # 1. Classical self‑attention
        attn_out = self.attention.run(rotation_params, entangle_params, inputs)
        attn_tensor = torch.from_numpy(attn_out).float()

        # 2. LSTM expects (seq_len, batch, input_dim)
        seq_len, batch, _ = attn_tensor.shape[1], attn_tensor.shape[0], attn_tensor.shape[2]
        lstm_input = attn_tensor.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        lstm_out, _ = self.lstm(lstm_input)  # (seq_len, batch, hidden_dim)

        # 3. Classifier
        lstm_out = lstm_out.permute(1, 0, 2)  # back to (batch, seq_len, hidden_dim)
        flat = lstm_out.reshape(-1, self.hidden_dim)
        logits = self.fc(flat)
        logits = logits.reshape(batch, seq_len, -1)
        return logits.numpy()


__all__ = ["SelfAttentionHybrid"]
