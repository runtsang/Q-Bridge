import numpy as np
import torch
from torch import nn
from typing import Iterable, Tuple

class ClassicalSelfAttention:
    """Classical self‑attention block that mimics the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class QLSTM(nn.Module):
    """Classical LSTM cell that serves as a placeholder for quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class UnifiedQuantumLayer(nn.Module):
    """Classic‑only implementation of the hybrid layer."""
    def __init__(self, n_features: int = 1, embed_dim: int = 4, n_qubits: int = 4, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.fc = nn.Linear(n_features, 1, bias=True)
        self.attention = ClassicalSelfAttention(embed_dim)
        self.lstm_gate = QLSTM(input_dim=embed_dim, hidden_dim=1, n_qubits=n_qubits)

    def run(self, mode: str, *args, **kwargs):
        """
        Dispatch to sub‑modules.

        Parameters
        ----------
        mode : str
            One of ``'fc'``, ``'attention'`` or ``'lstm'``.
        """
        if mode == "fc":
            return self.fc_forward(*args, **kwargs)
        elif mode == "attention":
            return self.attention.run(*args, **kwargs)
        elif mode == "lstm":
            return self.lstm_gate.forward(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def fc_forward(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.fc(values)).mean(dim=0)
        return expectation.detach().numpy()
