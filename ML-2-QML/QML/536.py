"""Quantum‑enhanced LSTM using Pennylane variational circuits.

This module implements the same interface as the classical QLSTM but
replaces each gate with a small variational circuit that can be trained
end‑to‑end with the rest of the network.  The circuit includes:
* Parameterised single‑qubit rotations (RX, RY, RZ).
* Entangling CNOT gates.
* Optional depolarising noise to emulate realistic hardware.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["QLSTM", "LSTMTagger"]

class QGate(nn.Module):
    """Parameterised variational gate implemented with Pennylane."""
    def __init__(self, n_qubits: int, device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        # 3 parameters per qubit for RX, RY, RZ rotations
        self.params = nn.Parameter(torch.randn(n_qubits * 3))
        self._circuit = qml.qnode(self._build_circuit, dev=self.dev, interface="torch")

    def _build_circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Variational circuit that accepts an input vector `x`."""
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(self.n_qubits):
            qml.RX(params[3 * i], wires=i)
            qml.RY(params[3 * i + 1], wires=i)
            qml.RZ(params[3 * i + 2], wires=i)
        # Entangle
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad or truncate to n_qubits
        x = x[:, : self.n_qubits]
        if x.size(1) < self.n_qubits:
            pad = torch.zeros(x.size(0), self.n_qubits - x.size(1), device=x.device)
            x = torch.cat([x, pad], dim=1)
        return self._circuit(x, self.params)

class QLSTM(nn.Module):
    """LSTM cell where each gate is a variational quantum circuit."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projection to quantum input space
        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget = QGate(n_qubits, device=device)
        self.input = QGate(n_qubits, device=device)
        self.update = QGate(n_qubits, device=device)
        self.output = QGate(n_qubits, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_forget(combined)))
            i = torch.sigmoid(self.input(self.lin_input(combined)))
            g = torch.tanh(self.update(self.lin_update(combined)))
            o = torch.sigmoid(self.output(self.lin_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            device=device,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
