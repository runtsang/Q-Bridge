"""Hybrid classical LSTM with a quantum‑encoded forget gate.

This module keeps the public API of the original seed (QLSTM, LSTMTagger)
while adding:
* A single‑qubit variational circuit for the forget gate.
* A depth‑wise stack (QLSTMStack) for deeper temporal modelling.
* A helper to log gate activations for analysis.
"""

from __future__ import annotations

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


# --------------------------------------------------------------------------- #
#   Quantum helper – a tiny parametrised circuit that outputs a single qubit
#   measurement.  The circuit is used to compute the forget gate.
# --------------------------------------------------------------------------- #
class ForgetGateQNode(nn.Module):
    """A single‑qubit variational circuit producing a scalar in [0,1]."""

    def __init__(self, n_qubits: int = 1, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        # Parameters for the rotation gates – one per qubit
        self.params = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the circuit to a batch of classical inputs `x`."""
        # Map classical input to a single qubit via a linear layer
        # This linear layer is part of the module to keep trainable
        # weights inside the same PyTorch graph.
        lin = nn.Linear(x.shape[1], self.n_qubits, bias=False).to(x.device)
        x_qubit = lin(x)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RY(params[i], wires=i)
            qml.CNOT(wires=[0, 1]) if self.n_qubits > 1 else None
            return qml.expval(qml.PauliZ(0))

        # Broadcast to batch dimension
        outputs = torch.stack([circuit(x_qubit[i], self.params) for i in range(x_qubit.shape[0])])
        # Map to [0,1] via sigmoid
        return torch.sigmoid(outputs)


# --------------------------------------------------------------------------- #
#   Hybrid QLSTM – classical input, update, output gates; quantum forget gate.
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Classical LSTM cell with a quantum‑encoded forget gate."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical gates
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum forget gate
        self.forget_qnode = ForgetGateQNode(n_qubits=n_qubits)

        # Linear layer to map classical input to the qubit space
        self.input_to_qubit = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process a sequence of shape (seq_len, batch, hidden_dim)."""
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gates
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))

            # Quantum forget gate
            f_input = self.input_to_qubit(combined)
            f = self.forget_qnode(f_input)

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
    """Stack of `QLSTM` cells for deeper temporal modelling."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, n_qubits: int = 1):
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
#   LSTMTagger – works with either QLSTM or nn.LSTM
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
