"""
Quantum implementation of the hybrid LSTM.  Each gate is a PennyLane
variational circuit that returns a vector of Pauli‑Z expectation values.
The interface matches the classical counterpart so that the same
tagger can be used for ablation studies.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QGate(nn.Module):
    """Variational quantum circuit acting as a gate.

    The circuit encodes the input vector as RX rotations, applies ``depth``
    layers of Ry,Rz,Rx rotations, entangles the qubits with CNOT chains,
    and returns the vector of Pauli‑Z expectation values.
    """

    def __init__(self, n_qubits: int, depth: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device("default.qubit", wires=n_qubits)
        # Parameters: shape (depth, n_qubits, 3) for (Ry,Rz,Rx)
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 3))

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(params, x):
            # Input encoding
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            # Variational layers
            for d in range(depth):
                for i in range(n_qubits):
                    qml.RY(params[d][i][0], wires=i)
                    qml.RZ(params[d][i][1], wires=i)
                    qml.RX(params[d][i][2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Return vector of Pauli‑Z expectation values
            expvals = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            return torch.stack(expvals)

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch, n_qubits)
        batch_outputs = []
        for xi in x:
            batch_outputs.append(self.qnode(self.params, xi))
        return torch.stack(batch_outputs)


class QLSTM(nn.Module):
    """Hybrid LSTM where gates are realised by PennyLane quantum circuits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        projection_dim: int = 64,
        *,
        depth: int = 1,
        return_all: bool = False,
        share_gates: bool = False,
        quantum_gate_only: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.projection_dim = projection_dim
        self.depth = depth
        self.return_all = return_all
        self.share_gates = share_gates
        self.quantum_gate_only = quantum_gate_only

        self.shared_proj = nn.Linear(input_dim + hidden_dim, projection_dim)
        self.linear_to_qubits = nn.Linear(projection_dim, n_qubits, bias=False)

        if share_gates:
            gate = nn.ModuleList([QGate(n_qubits, depth) for _ in range(depth)])
            self.forget_gate = self.input_gate = self.update_gate = self.output_gate = gate
        else:
            self.forget_gate = nn.ModuleList([QGate(n_qubits, depth) for _ in range(depth)])
            self.input_gate = nn.ModuleList([QGate(n_qubits, depth) for _ in range(depth)])
            self.update_gate = nn.ModuleList([QGate(n_qubits, depth) for _ in range(depth)])
            self.output_gate = nn.ModuleList([QGate(n_qubits, depth) for _ in range(depth)])

        self.linear_qubit_to_hidden = nn.Linear(n_qubits, hidden_dim, bias=False)

    def _apply_gate(self, gate_list: nn.ModuleList, vec: torch.Tensor) -> torch.Tensor:
        for gate in gate_list:
            vec = gate(vec)
        return vec

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            proj = self.shared_proj(combined)
            qinput = combined if self.quantum_gate_only else self.linear_to_qubits(proj)
            f = self._apply_gate(self.forget_gate, qinput)
            i = self._apply_gate(self.input_gate, qinput)
            g = self._apply_gate(self.update_gate, qinput)
            o = self._apply_gate(self.output_gate, qinput)
            f = torch.sigmoid(self.linear_qubit_to_hidden(f))
            i = torch.sigmoid(self.linear_qubit_to_hidden(i))
            g = torch.tanh(self.linear_qubit_to_hidden(g))
            o = torch.sigmoid(self.linear_qubit_to_hidden(o))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        if self.return_all:
            return stacked, (hx, cx)
        return stacked[-1], (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence‑tagging model that can switch between the quantum LSTM and
    a standard `nn.LSTM`.  The API is identical to the original seed.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits,
                **kwargs,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
