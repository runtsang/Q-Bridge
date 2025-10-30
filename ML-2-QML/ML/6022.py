"""UnifiedSamplerQLSTM: Classical core with optional quantum modules.

This module defines a lightweight sampler network and a classical LSTM that can be
replaced by a quantum LSTM when `n_qubits > 0`. The two components are
connected end‑to‑end so that the sampler output becomes the input to the LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerModule(nn.Module):
    """
    Simple two‑layer linear sampler.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    hidden_dim : int
        Size of the hidden layer.
    output_dim : int
        Dimension of the sampler output (typically equal to hidden_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class QLSTM(nn.Module):
    """
    Classical LSTM cell with linear gates, mirroring the quantum interface.
    """
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
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if n_qubits > 0:
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # inputs shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(inputs.permute(1,0,2))
        tag_logits = self.hidden2tag(lstm_out.permute(1,0,2))
        return F.log_softmax(tag_logits, dim=2)

class UnifiedSamplerQLSTM(nn.Module):
    """
    Hybrid sampler‑LSTM module.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw feature vector fed to the sampler.
    hidden_dim : int
        Dimensionality of the sampler output and the hidden state of the LSTM.
    tagset_size : int
        Number of target tags.
    n_qubits : int, optional
        If >0, quantum LSTM gates are used; otherwise classical LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.sampler = SamplerModule(input_dim, hidden_dim, output_dim=hidden_dim)
        self.tagger = LSTMTagger(hidden_dim, hidden_dim, tagset_size, n_qubits=n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, seq_len, input_dim)
        batch, seq_len, _ = inputs.shape
        flat_inputs = inputs.view(batch*seq_len, -1)
        probs = self.sampler(flat_inputs)
        sampled = probs.view(batch, seq_len, -1)
        return self.tagger(sampled)

__all__ = ["UnifiedSamplerQLSTM"]
