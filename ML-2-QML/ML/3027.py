"""
Hybrid LSTM implementation with optional quantum gates and a classical
sampler network.

This module is intentionally lightweight and fully classical.  It
provides a drop‑in replacement for the original QLSTM, but it can
optionally delegate gate computation to a quantum module that lives
in :mod:`qml`.  Importing the quantum module is deferred until it
is actually needed, so the pure‑Python path remains fast and
dependency‑free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Lazy import of the quantum gate – the heavy dependency is only
# pulled in if the user explicitly requests a quantum LSTM.
try:
    from.qml import QGate  # type: ignore
except Exception:  # pragma: no cover
    QGate = None  # type: ignore


class ClassicQLSTM(nn.Module):
    """
    Purely classical LSTM cell.

    Parameters
    ----------
    input_dim : int
        Dimension of each input token.
    hidden_dim : int
        Number of hidden units.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )


class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM that forwards each gate through a quantum module.

    Parameters
    ----------
    input_dim : int
        Input token dimension.
    hidden_dim : int
        Hidden state dimension.
    n_qubits : int
        Number of qubits used by the quantum gate.  If ``0`` the
        module silently falls back to the classical implementation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        if self.n_qubits > 0 and QGate is not None:
            # Quantum gate is available – use it for all four LSTM gates.
            self.forget_gate = QGate(n_qubits)
            self.input_gate = QGate(n_qubits)
            self.update_gate = QGate(n_qubits)
            self.output_gate = QGate(n_qubits)
            # Linear maps that feed quantum circuits.
            self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # Fallback to the classical implementation.
            self.forget_gate = ClassicQLSTM
            self.input_gate = ClassicQLSTM
            self.update_gate = ClassicQLSTM
            self.output_gate = ClassicQLSTM

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
                i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
                g = torch.tanh(self.update_gate(self.update_lin(combined)))
                o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            else:
                f = torch.sigmoid(self.forget_gate(forget_lin=combined))
                i = torch.sigmoid(self.input_gate(input_lin=combined))
                g = torch.tanh(self.update_gate(update_lin=combined))
                o = torch.sigmoid(self.output_gate(output_lin=combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )


class SamplerQNN(nn.Module):
    """
    Classical approximation of a quantum sampler network.

    This small network mimics the output distribution of a
    parameterised quantum sampler.  It is useful when a full
    quantum backend is unavailable but a probabilistic gating
    mechanism is still desired.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classic and hybrid LSTM.

    Parameters
    ----------
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Hidden state size.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of output tags.
    n_qubits : int, default 0
        If >0 a hybrid LSTM with quantum gates is used.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = ClassicQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)


__all__ = ["ClassicQLSTM", "HybridQLSTM", "SamplerQNN", "LSTMTagger"]
