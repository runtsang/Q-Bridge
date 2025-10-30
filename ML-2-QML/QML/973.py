"""Quantum‑enhanced LSTM layer using a parameter‑efficient ansatz."""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple, Optional

__all__ = ["QLSTMGen"]


class QGate(nn.Module):
    """Parameter‑efficient quantum gate that outputs a probability vector."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.params = nn.ParameterList(
            [nn.Parameter(torch.rand(1)) for _ in range(n_qubits)]
        )
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None, interface="torch")

        def circuit(x: torch.Tensor):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
                qml.RY(self.params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qml.QNode(circuit, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)


class QLSTMQuantum(nn.Module):
    """LSTM cell where each gate is realized by a small quantum circuit."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QGate(n_qubits)
        self.input_gate = QGate(n_qubits)
        self.update_gate = QGate(n_qubits)
        self.output_gate = QGate(n_qubits)

        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class QLSTMGen(nn.Module):
    """Wrapper that selects quantum or classical LSTM based on n_qubits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        if n_qubits > 0:
            self.lstm = QLSTMQuantum(input_dim, hidden_dim, n_qubits)
        else:
            raise ValueError("n_qubits must be > 0 for quantum mode")

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(inputs, states)
