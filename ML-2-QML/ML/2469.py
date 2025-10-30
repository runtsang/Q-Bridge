"""Hybrid classical LSTM with fraud‑detection style parameter clipping and scaling.

The module defines a `QLSTM` that behaves like a standard PyTorch LSTM but
allows optional weight clipping and output scaling, inspired by the fraud
detection example.  The accompanying `LSTMTagger` retains the original
sequence‑tagging interface.

The implementation keeps the same public API as the original `QLSTM.py`
so that existing pipelines can switch to this module without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters used to initialise a linear gate in a classical LSTM."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters, *, clip: bool, scale: bool
) -> nn.Module:
    """Return a linear gate with optional clipping and scaling."""
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
    scale_tensor = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift_tensor = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale_tensor)
            self.register_buffer("shift", shift_tensor)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class QLSTM(nn.Module):
    """Classical LSTM cell with optional fraud‑detection style clipping and scaling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        clip: bool = False,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.clip = clip
        self.scale = scale

        # Use the same gate layout as the original QLSTM
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if clip:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.clamp_(-5.0, 5.0)
                    m.bias.data.clamp_(-5.0, 5.0)

        # Optional per‑gate scaling buffers
        if scale:
            self.forget_scale = nn.Parameter(torch.ones(hidden_dim))
            self.input_scale = nn.Parameter(torch.ones(hidden_dim))
            self.update_scale = nn.Parameter(torch.ones(hidden_dim))
            self.output_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            if self.scale:
                f *= self.forget_scale
                i *= self.input_scale
                g *= self.update_scale
                o *= self.output_scale
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the classical QLSTM or a standard LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        clip: bool = False,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                clip=clip,
                scale=scale,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(
            embeds.view(len(sentence), 1, -1)
        )
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
