"""Quantum‑enhanced LSTM using Pennylane variational circuits.

This module mirrors the classical‑quantum hybrid from the ML side but uses
Pennylane for the quantum back‑end.  The public API remains unchanged.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class _VarGatePN(nn.Module):
    """Variational gate built with Pennylane.

    The circuit depth can be increased to add expressivity.
    """

    def __init__(self, in_dim: int, n_wires: int, depth: int = 1, dev_type: str = "default.qubit"):
        super().__init__()
        self.in_dim = in_dim
        self.n_wires = n_wires
        self.depth = depth
        self.dev_type = dev_type

        self.lin = nn.Linear(in_dim, n_wires, bias=False)
        self.params = nn.Parameter(torch.randn(depth, n_wires, 1))
        self.last_expect = None

        self.dev = qml.device(dev_type, wires=n_wires)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x_angles, params):
            # Encode classical data via RX
            for wire in range(n_wires):
                qml.RX(x_angles[wire], wires=wire)
            # Variational layers
            for d in range(depth):
                for wire in range(n_wires):
                    qml.RX(params[d, wire, 0], wires=wire)
                # CNOT chain
                for wire in range(n_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            # Output expectation values of PauliZ
            return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        angles = self.lin(x)  # (batch, n_wires)
        out = []
        for i in range(batch_size):
            out.append(self.circuit(angles[i], self.params))
        out = torch.stack(out, dim=0)
        self.last_expect = out
        return out


class QLSTM(nn.Module):
    """Hybrid LSTM cell, quantum gates implemented with Pennylane."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        if n_qubits > 0:
            assert hidden_dim == n_qubits, "hidden_dim must equal n_qubits for quantum mode"
            self.forget = _VarGatePN(input_dim + hidden_dim, n_qubits, depth)
            self.input = _VarGatePN(input_dim + hidden_dim, n_qubits, depth)
            self.update = _VarGatePN(input_dim + hidden_dim, n_qubits, depth)
            self.output = _VarGatePN(input_dim + hidden_dim, n_qubits, depth)
            self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.lin_forget(combined)))
                i = torch.sigmoid(self.input(self.lin_input(combined)))
                g = torch.tanh(self.update(self.lin_update(combined)))
                o = torch.sigmoid(self.output(self.lin_output(combined)))
            else:
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def get_expectations(self) -> dict[str, torch.Tensor]:
        if self.n_qubits == 0:
            raise RuntimeError("No quantum gates present")
        return {
            "forget": self.forget.last_expect,
            "input": self.input.last_expect,
            "update": self.update.last_expect,
            "output": self.output.last_expect,
        }


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
