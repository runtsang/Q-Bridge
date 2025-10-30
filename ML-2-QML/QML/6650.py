"""Quantum‑enhanced LSTM where all gates share a single variational circuit."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLayer(tq.QuantumModule):
    """
    Parameterised quantum circuit that accepts an n‑qubit input tensor and returns
    a measurement vector.  All rotation angles are trainable and are shared across
    the four LSTM gates, dramatically reducing the number of quantum parameters.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Trainable rotation angles (one per wire)
        self.params = nn.Parameter(torch.randn(n_wires))
        # Entanglement pattern (cyclic CNOT chain)
        self.cnot_pattern = [(i, (i + 1) % n_wires) for i in range(n_wires)]
        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    @property
    def encoder(self) -> tq.GeneralEncoder:
        """Lazy‑initialised encoder that maps a batch of rotation angles to the device."""
        if not hasattr(self, "_encoder"):
            self._encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
        return self._encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_wires) containing rotation angles.
        Returns:
            Tensor of shape (batch, n_wires) with measurement outcomes.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        # Apply the shared parametrised rotations
        for wire in range(self.n_wires):
            tqf.rx(qdev, wires=wire, params=self.params[wire])
        # Entangle the wires
        for wire, target in self.cnot_pattern:
            tqf.cnot(qdev, wires=[wire, target])
        return self.measure(qdev)

class QLSTMGen(nn.Module):
    """
    Quantum‑LSTM that re‑uses a single variational circuit for all four gates.
    The input and hidden state are linearly projected to `n_qubits` qubits,
    then fed through the shared `QLayer`.  Gate activations are applied
    exactly as in the classical version.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to the qubit space
        self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits, bias=False)

        # Shared quantum layer
        self.qlayer = QLayer(n_qubits)

        # Linear layers to map the quantum output back to gate space
        self.gate_mapper = nn.Linear(n_qubits, 4 * hidden_dim, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            proj = self.input_proj(combined)          # (batch, n_qubits)
            q_out = self.qlayer(proj)                 # (batch, n_qubits)
            gate_vals = self.gate_mapper(q_out)       # (batch, 4*hidden_dim)
            f, i, g, o = gate_vals.chunk(4, dim=1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
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
    """Sequence tagging model that uses the quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTagger"]
