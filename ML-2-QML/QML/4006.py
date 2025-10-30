"""Quantum‑enhanced LSTM cell that fuses small quantum estimators with classical gates.

The implementation borrows the gate‑by‑gate architecture of a classical
LSTM but replaces each gate with a single‑qubit parameterised circuit
that mimics the behaviour of a tiny EstimatorQNN.  This provides a
fully quantum‑centric building block that can be stacked into a
sequence‑tagging network.

Classes
-------
CombinedQLSTM
    Quantum LSTM cell where each gate is a 1‑qubit estimator circuit.
LSTMTagger
    Wrapper that applies the cell to a sequence and projects to tags.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayerQuantum(tq.QuantumModule):
    """Single‑qubit estimator circuit inspired by EstimatorQNN."""
    def __init__(self):
        super().__init__()
        # Trainable weight for the RX gate
        self.weight = nn.Parameter(torch.rand(1) * 2 - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expectation value of Pauli‑Z after H‑Ry‑Rx circuit
        batch_size = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=1, bsz=batch_size, device=x.device)

        # Encode the first input feature with a Ry rotation
        tq.RY(x[:, 0], wires=0)(qdev)
        # Parameterised RX rotation
        tq.RX(self.weight, wires=0)(qdev)
        # Hadamard to switch measurement basis
        tq.H(wires=0)(qdev)

        # Measure expectation of Z
        exp_z = tq.Expectation(tq.PauliZ, wires=0)
        return exp_z(qdev)


class CombinedQLSTM(nn.Module):
    """Quantum LSTM cell where each gate is a 1‑qubit estimator."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget_gate = QLayerQuantum()
        self.input_gate = QLayerQuantum()
        self.update_gate = QLayerQuantum()
        self.output_gate = QLayerQuantum()

        # Linear layers map the concatenated (x,h) to a scalar
        self.forget_linear = nn.Linear(input_dim + hidden_dim, 1)
        self.input_linear = nn.Linear(input_dim + hidden_dim, 1)
        self.update_linear = nn.Linear(input_dim + hidden_dim, 1)
        self.output_linear = nn.Linear(input_dim + hidden_dim, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))

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


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses a quantum LSTM cell."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = CombinedQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["CombinedQLSTM", "LSTMTagger"]
