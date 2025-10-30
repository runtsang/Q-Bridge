"""Quantum LSTM with PennyLane variational circuits.

Each gate (forget, input, update, output) is implemented as a
variational circuit that takes the linear projection of the
concatenated input and hidden state as classical parameters.
The circuit returns a vector of expectation values that is
passed through the same non‑linearity as in the classical
implementation.  The module is fully differentiable and can
be trained with the same optimisers as a normal PyTorch model.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QGate(nn.Module):
    """Variational gate implemented with PennyLane.

    The gate encodes the classical input into RX rotations,
    applies trainable RY rotations, entangles the wires with
    a linear CNOT chain and measures Pauli‑Z expectation values.
    """

    def __init__(
        self,
        n_wires: int,
        dev_name: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device(dev_name, wires=n_wires, shots=shots)
        # Trainable parameters for the variational layer
        self.params = nn.Parameter(torch.randn(n_wires))
        # QNode with Torch interface for autograd support
        self.qnode = qml.QNode(
            self._circuit, self.dev, interface="torch", diff_method="backprop"
        )

    def _circuit(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Encode classical input
        for i in range(self.n_wires):
            qml.RX(x[i], wires=i)
        # Variational RY rotations
        for i in range(self.n_wires):
            qml.RY(params[i], wires=i)
        # Simple linear entanglement
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        # Measure expectation values of Pauli‑Z
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the circuit for each sample in the batch."""
        batch_size = x.shape[0]
        out = []
        for i in range(batch_size):
            out.append(self.qnode(self.params, x[i]))
        return torch.stack(out)


class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell.

    Each gate is a ``QGate`` that receives a linear projection of
    the concatenated input and hidden state.  The output of the
    gate is passed through the same activation functions as the
    classical LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dev_name: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to the number of qubits
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate = QGate(n_qubits, dev_name, shots)
        self.input_gate = QGate(n_qubits, dev_name, shots)
        self.update_gate = QGate(n_qubits, dev_name, shots)
        self.output_gate = QGate(n_qubits, dev_name, shots)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
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
    """Sequence tagging model that can switch between the quantum and
    classical LSTM implementations.
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


__all__ = ["QLSTM", "LSTMTagger"]
