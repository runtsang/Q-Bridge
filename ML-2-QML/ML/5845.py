"""Hybrid classical LSTM with optional clipping and quantum‑style gate wrappers.

The module re‑implements the original QLSTM but extends it with:
* parameter clipping inspired by the fraud‑detection example,
* a flag that selects between a fully classical linear gate or a
  lightweight quantum‑style wrapper that simply masks the output,
* a unified `QLSTMGen` class that can be dropped into any sequence
  tagging pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Parameter utilities (borrowed from FraudDetection)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


# --------------------------------------------------------------------------- #
#  Gate wrapper – can be classical or quantum‑style
# --------------------------------------------------------------------------- #

class _GateWrapper(nn.Module):
    """Wrapper that can be either a classical linear gate or a dummy quantum gate."""
    def __init__(self, in_features: int, out_features: int, use_quantum: bool):
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            # Quantum gates are simulated by a linear layer followed by a tanh
            # that mimics the output of a measurement with bounded range.
            self.linear = nn.Linear(in_features, out_features)
            self.activation = nn.Tanh()
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            return self.activation(self.linear(x))
        return self.linear(x)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM cell
# --------------------------------------------------------------------------- #

class QLSTMGen(nn.Module):
    """Hybrid LSTM cell that can use classical or quantum‑style gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, use_quantum: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        gate_dim = hidden_dim
        self.forget = _GateWrapper(input_dim + hidden_dim, gate_dim, use_quantum)
        self.input = _GateWrapper(input_dim + hidden_dim, gate_dim, use_quantum)
        self.update = _GateWrapper(input_dim + hidden_dim, gate_dim, use_quantum)
        self.output = _GateWrapper(input_dim + hidden_dim, gate_dim, use_quantum)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  Sequence tagging model
# --------------------------------------------------------------------------- #

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the hybrid LSTM and nn.LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, use_quantum)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen", "LSTMTagger", "FraudLayerParameters", "_layer_from_params"]
