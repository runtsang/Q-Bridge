"""Hybrid quantum‑classical LSTM (quantum backbone).

This module implements the same public API as the classical version
but replaces each LSTM gate with a small variational quantum circuit.
The quantum layer is built on top of torchquantum and can be used
for experiments that compare classical and quantum LSTM behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


@dataclass
class HybridQLSTMConfig:
    """Configuration for the hybrid LSTM."""
    input_dim: int
    hidden_dim: int
    n_qubits: int
    batch_size: int = 32
    dropout: float = 0.0


class QuantumGate(tq.QuantumModule):
    """Small variational circuit that implements a single LSTM gate."""

    def __init__(self, n_wires: int, n_params: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_params = n_params

        # Encoder: a single RX gate per wire that takes the classical
        # input vector as parameters.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )

        # Parameterised rotations that will be trained.
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        # Measurement in the Pauli‑Z basis.
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum gate.

        Args:
            x: Tensor of shape (batch, n_wires) containing the
               gate‑specific linear output.
        Returns:
            Tensor of shape (batch, n_wires) containing the measurement
            results (±1) that will be fed into the classical LSTM cell.
        """
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Simple CNOT chain to create entanglement
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class HybridQLSTM(nn.Module):
    """Quantum LSTM cell that mirrors the classical interface."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        mode: str = "quantum",
    ) -> None:
        super().__init__()
        if mode!= "quantum":
            raise ValueError("Quantum implementation only supports mode='quantum'")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget = QuantumGate(n_qubits)
        self.input = QuantumGate(n_qubits)
        self.update = QuantumGate(n_qubits)
        self.output = QuantumGate(n_qubits)

        # Linear layers mapping the classical concatenated input to qubits
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.forget_lin(combined)))
            i = torch.sigmoid(self.input(self.input_lin(combined)))
            g = torch.tanh(self.update(self.update_lin(combined)))
            o = torch.sigmoid(self.output(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

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


class HybridTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        mode: str = "quantum",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if mode == "classical" and n_qubits > 0:
            raise RuntimeError(
                "Classical mode is not available in the quantum module. "
                "Import the classical implementation to use a classical LSTM."
            )
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits, mode="quantum")
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridTagger", "HybridQLSTMConfig"]
