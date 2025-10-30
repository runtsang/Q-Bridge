"""Quantum‑enhanced LSTM using Pennylane for all gates.

The module keeps the same public API as the original seed but replaces the
classical gates with quantum circuits.  Each gate is a variational circuit
that outputs a probability in [0,1] via measurement of a single qubit.
"""

from __future__ import annotations

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


# --------------------------------------------------------------------------- #
#   Helper – variational circuit for a single gate
# --------------------------------------------------------------------------- #
class GateCircuit(nn.Module):
    """Variational circuit that implements a single LSTM gate."""

    def __init__(self, n_qubits: int, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        # Parameters for rotation gates – one per qubit
        self.params = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to classical input `x`."""
        # Map classical input to qubit amplitudes via a linear layer
        lin = nn.Linear(x.shape[1], self.n_qubits, bias=False).to(x.device)
        x_qubit = lin(x)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RY(params[i], wires=i)
            # Simple entangling pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        outputs = torch.stack([circuit(x_qubit[i], self.params) for i in range(x_qubit.shape[0])])
        return torch.sigmoid(outputs)  # map to [0,1]


# --------------------------------------------------------------------------- #
#   Fully quantum QLSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Fully quantum LSTM cell – all gates are quantum circuits."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.input_gate = GateCircuit(n_qubits)
        self.forget_gate = GateCircuit(n_qubits)
        self.update_gate = GateCircuit(n_qubits)
        self.output_gate = GateCircuit(n_qubits)

        # Linear mapping from classical input to qubit space
        self.input_to_qubit = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Map to qubit space
            q_input = self.input_to_qubit(combined)

            # Quantum gates
            i = self.input_gate(q_input)
            f = self.forget_gate(q_input)
            g = self.update_gate(q_input)
            o = self.output_gate(q_input)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#   Depth‑wise stack of QLSTM cells
# --------------------------------------------------------------------------- #
class QLSTMStack(nn.Module):
    """Stack of `QLSTM` cells for deeper quantum temporal modelling."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, n_qubits: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                QLSTM(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    n_qubits=n_qubits,
                )
                for i in range(n_layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        layer_states = states if states is not None else [None] * len(self.layers)
        outputs = inputs
        next_states = []

        for layer, s in zip(self.layers, layer_states):
            outputs, s = layer(outputs, states=s)
            next_states.append(s)

        return outputs, next_states


# --------------------------------------------------------------------------- #
#   LSTMTagger – works with either quantum or classical LSTM
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTMStack(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_qubits=n_qubits,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "QLSTMStack", "LSTMTagger"]
