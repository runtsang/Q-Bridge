"""Quantum‑enhanced LSTM layers for sequence tagging using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


@dataclass
class QLSTMConfig:
    """Configuration for the quantum part of the QLSTM."""
    n_wires: int = 12          # number of qubits in the circuit
    depth: int = 2            # depth of the variational block
    random_seed: int | None = None  # for reproducibility


class QGate(nn.Module):
    """Quantum gate that maps a classical vector to a quantum measurement."""

    def __init__(self, n_wires: int, depth: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # trainable variational parameters
        self.theta = nn.Parameter(torch.randn(depth, n_wires))
        self.dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inp: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            # Feature encoding: each input component drives an RX rotation
            for i in range(self.n_wires):
                qml.RX(inp[i], wires=i)
            # Variational block
            for d in range(self.depth):
                for i in range(self.n_wires):
                    qml.RY(theta[d, i], wires=i)
                # Entangling layer (ring of CNOTs)
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_wires - 1, 0])
            # Measure expectation values of Pauli‑Z on every qubit
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)])

        self.circuit = circuit

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum circuit for a batch of inputs.

        Args:
            inp: Tensor of shape ``(batch, n_wires)``.

        Returns:
            Tensor of shape ``(batch, n_wires)`` containing the measurement
            results for each qubit.
        """
        batch = inp.shape[0]
        out = []
        for i in range(batch):
            out.append(self.circuit(inp[i], self.theta))
        return torch.stack(out)


class QLSTM(nn.Module):
    """Drop‑in replacement for the original QLSTM, now employing a
    variational quantum circuit for each gate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        config: QLSTMConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.config = config or QLSTMConfig()

        # Linear projections to the number of qubits
        self.forget_lin = nn.Linear(input_dim + hidden_dim, self.config.n_wires)
        self.input_lin = nn.Linear(input_dim + hidden_dim, self.config.n_wires)
        self.update_lin = nn.Linear(input_dim + hidden_dim, self.config.n_wires)
        self.output_lin = nn.Linear(input_dim + hidden_dim, self.config.n_wires)

        # Quantum gates
        self.forget_gate = QGate(self.config.n_wires, self.config.depth)
        self.input_gate = QGate(self.config.n_wires, self.config.depth)
        self.update_gate = QGate(self.config.n_wires, self.config.depth)
        self.output_gate = QGate(self.config.n_wires, self.config.depth)

        # Projection from quantum output back to hidden dimension
        self.forget_proj = nn.Linear(self.config.n_wires, hidden_dim)
        self.input_proj = nn.Linear(self.config.n_wires, hidden_dim)
        self.update_proj = nn.Linear(self.config.n_wires, hidden_dim)
        self.output_proj = nn.Linear(self.config.n_wires, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the hybrid quantum‑classical LSTM.

        Args:
            inputs: Tensor of shape ``(seq_len, batch, input_dim)``.
            states: Optional tuple ``(hx, cx)``.

        Returns:
            Tuple of the output sequence and the final hidden & cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(
                self.forget_proj(
                    self.forget_gate(
                        self.forget_lin(combined)
                    )
                )
            )
            i = torch.sigmoid(
                self.input_proj(
                    self.input_gate(
                        self.input_lin(combined)
                    )
                )
            )
            g = torch.tanh(
                self.update_proj(
                    self.update_gate(
                        self.update_lin(combined)
                    )
                )
            )
            o = torch.sigmoid(
                self.output_proj(
                    self.output_gate(
                        self.output_lin(combined)
                    )
                )
            )
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
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        config: QLSTMConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, config=config)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
