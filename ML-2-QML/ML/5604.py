"""Unified classical kernel framework combining RBF, self‑attention and LSTM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Classical kernel utilities ------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --- Self‑attention helper ----------------------------------------------------- #

class SelfAttention:
    """Simple classical self‑attention block."""

    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --- Classical LSTM surrogate --------------------------------------------------- #

class QLSTM(nn.Module):
    """Linear‑gate LSTM surrogate."""

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --- Unified interface ---------------------------------------------------------- #

class QuantumKernelMethod:
    """
    Classical surrogate for a quantum‑style kernel engine.
    Supports:
        * RBF kernel on raw vectors
        * Optional classical self‑attention pre‑processing
        * Optional classical LSTM feature extraction for sequences
    """

    def __init__(
        self,
        mode: str = "rbf",
        gamma: float = 1.0,
        attention: bool = False,
        lstm: bool = False,
        n_qubits: int = 4,
    ) -> None:
        self.mode = mode
        self.gamma = gamma
        self.attention = attention
        self.lstm = lstm
        self.n_qubits = n_qubits

        self.kernel = Kernel(gamma)
        self.attention_module = SelfAttention() if attention else None
        self.lstm_module = QLSTM(4, 4, n_qubits) if lstm else None

    def _apply_attention(self, X: torch.Tensor) -> torch.Tensor:
        if self.attention_module is None:
            return X
        rot = np.random.randn(X.shape[1], 12)
        ent = np.random.randn(X.shape[1], 4)
        return self.attention_module.run(rot, ent, X.numpy())

    def _apply_lstm(self, seqs: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.lstm_module is None:
            return seqs
        hidden_states = []
        for seq in seqs:
            seq = seq.unsqueeze(1)  # batch dimension
            out, _ = self.lstm_module(seq)
            hidden_states.append(out.squeeze(1))
        return hidden_states

    def compute_kernel(
        self,
        a: list[torch.Tensor],
        b: list[torch.Tensor],
        data_type: str = "vector",
    ) -> np.ndarray:
        """
        Compute a Gram matrix between two collections of data.
        * vector – pair‑wise RBF
        * sequence – average RBF over hidden states
        """
        if data_type == "vector":
            a = self._apply_attention(a)
            b = self._apply_attention(b)
            return kernel_matrix(a, b, gamma=self.gamma)

        if data_type == "sequence":
            a_hidden = self._apply_lstm(a)
            b_hidden = self._apply_lstm(b)
            K = np.zeros((len(a_hidden), len(b_hidden)))
            for i, ha in enumerate(a_hidden):
                for j, hb in enumerate(b_hidden):
                    mat = kernel_matrix(ha, hb, gamma=self.gamma)
                    K[i, j] = mat.mean()
            return K

        raise ValueError(f"Unsupported data type: {data_type}")

__all__ = ["QuantumKernelMethod"]
