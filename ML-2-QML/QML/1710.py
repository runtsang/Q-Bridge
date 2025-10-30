"""Quantum LSTM variant using PennyLane.

Each LSTM gate is implemented by a tiny variational circuit that
takes the linear projection of the concatenated input and hidden
state and produces a vector of size ``n_qubits``.  The circuit
encodes the input with RX rotations, applies a single layer of
parameterised RZ rotations and a chain of CNOTs, then returns the
Pauli‑Z expectation values of all qubits.  The output of the
circuit is fed back into the LSTM equations.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QLayer(nn.Module):
    """
    Variational quantum layer that maps an input vector of shape
    ``(batch, n_qubits)`` to an output of the same shape by measuring
    Pauli‑Z expectation values after a simple ansatz.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_qubits)])

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            for i in range(n_qubits):
                qml.RX(inputs[:, i], wires=i)
            for i in range(n_qubits):
                qml.RZ(self.params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``(batch, n_qubits)``

        Returns
        -------
        Tensor
            ``(batch, n_qubits)`` expectation values.
        """
        return self.circuit(x)


class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell.  The linear projections for each gate are
    identical to the classical version; the outputs of these projections are
    passed through a `QLayer` to obtain the gate values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Gate modules
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear projections from (input + hidden) to gate space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise hidden and cell states to zeros.
        """
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h0, c0

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : Tensor
            ``(seq_len, batch, input_dim)``
        states : Tuple[Tensor, Tensor] | None
            Optional external hidden and cell states.

        Returns
        -------
        outputs : Tensor
            ``(seq_len, batch, hidden_dim)``
        (h_n, c_n) : Tuple[Tensor, Tensor]
            Final hidden and cell states.
        """
        seq_len, batch_size, _ = inputs.shape
        device = inputs.device

        if states is None:
            hx, cx = self._init_states(batch_size, device)
        else:
            hx, cx = states

        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]  # (batch, input_dim)
            combined = torch.cat([x_t, hx], dim=1)

            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Apply dropout to the hidden state
            hx = self.dropout(hx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the classical
    `QLSTM` and the quantum `QLSTM` defined in this module.  The
    constructor signature matches the original seed but exposes a
    ``dropout`` argument for regularisation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
            )
        else:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                dropout=dropout if dropout > 0.0 else 0.0,
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.

        Parameters
        ----------
        sentence : Tensor
            ``(seq_len, batch)`` word indices.

        Returns
        -------
        Tensor
            Log‑softmaxed tag scores of shape ``(seq_len, batch, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM", "LSTMTagger"]
