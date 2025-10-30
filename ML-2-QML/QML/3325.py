"""Quantum‑enhanced LSTM with a parameterized sampler network.

The module introduces a `SamplerQNN` that implements the small two‑qubit
circuit used in the reference `SamplerQNN.py`.  Each LSTM gate is a
`QLayer` that wraps this sampler.  The sampler produces a two‑element
probability vector; the first component is used as the gate activation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class SamplerQNN(tq.QuantumModule):
    """
    Parameterised quantum sampler that mirrors the qiskit circuit from
    `SamplerQNN.py`.  It uses two qubits, applies Ry rotations for the
    input and weight parameters, entangles them with CX gates, and
    measures in the Pauli‑Z basis.
    """

    def __init__(self, n_wires: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that applies Ry gates for input parameters
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        # Trainable Ry gates for weight parameters
        self.params = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Entangling CX pattern matching the reference circuit
        self.cnot_pattern = [
            (0, 1),
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2) representing the two input
            parameters.

        Returns
        -------
        torch.Tensor
            Probabilities of measuring each computational basis state
            (shape (batch, 2)).
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for src, dst in self.cnot_pattern:
            tqf.cnot(qdev, wires=[src, dst])
        return self.measure(qdev)


class QLayer(tq.QuantumModule):
    """
    Gate‑level quantum layer that uses `SamplerQNN` to produce a
    probability distribution; the first probability is taken as the
    gate activation value.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.sampler = SamplerQNN(n_wires=n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.sampler(x)
        # Gate activation is the probability of measuring |0>
        return probs.select(1, 0)


class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell.  Each gate is a `QLayer` that uses a
    parameterised quantum sampler.  The linear projections convert the
    concatenated input/hidden vector into a 2‑dim vector suitable for
    the sampler.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum layers for each gate
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear projections to 2‑dim vectors for the sampler
        self.linear_forget = nn.Linear(input_dim + hidden_dim, 2)
        self.linear_input = nn.Linear(input_dim + hidden_dim, 2)
        self.linear_update = nn.Linear(input_dim + hidden_dim, 2)
        self.linear_output = nn.Linear(input_dim + hidden_dim, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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
    """
    Sequence tagging model that can switch between a classical LSTM and
    the quantum‑enhanced LSTM.
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
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "SamplerQNN"]
