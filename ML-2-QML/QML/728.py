"""Quantum‑enhanced hybrid LSTM.

This module implements the same public interface as the classical
module but replaces the placeholder `QuantumGate` with a real
variational circuit built with Pennylane.  The circuit operates on
`n_qubits` wires and returns a vector of expectation values that
is used as the gate values.  The design keeps the rest of the
cell identical to the classical version, enabling side‑by‑side
experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import pennylane as qml
import numpy as np

# --------------------------------------------------------------------------- #
# Quantum gate module
# --------------------------------------------------------------------------- #
class QuantumGate(nn.Module):
    """
    Variational quantum circuit that transforms an input vector of size
    `out_dim` into a new vector of the same size.  The circuit uses
    rotation gates on each qubit followed by a layer of CNOTs and
    measures the Pauli‑Z expectation value of every wire.
    """

    def __init__(self, out_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_qubits = n_qubits
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive for a quantum gate")

        # Device for the circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameter matrix that maps `out_dim` inputs to `n_qubits` rotation angles
        self.param_matrix = nn.Linear(out_dim, n_qubits, bias=False)

        # QNode with Torch interface
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            # x has shape (batch, n_qubits)
            for i in range(self.n_qubits):
                qml.RX(x[:, i], wires=i)
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Final CNOT to close the chain
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(wires=list(range(self.n_qubits))))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, out_dim).  The tensor is first linearly mapped to
            a vector of rotation angles of shape (batch, n_qubits) and then
            fed into the quantum circuit.
        Returns
        -------
        torch.Tensor
            Shape (batch, out_dim).  The circuit returns a vector of
            expectation values; it is linearly mapped back to the original
            dimension.
        """
        # Map to rotation angles
        angles = self.param_matrix(x)  # (batch, n_qubits)
        # Run the circuit
        z_expect = self.circuit(angles)  # (batch, n_qubits)
        # Map back to original dimension
        return self.param_matrix(z_expect)  # (batch, out_dim)


# --------------------------------------------------------------------------- #
# Hybrid LSTM cell using the real quantum gate
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM cell that uses the quantum `QuantumGate` defined above
    to augment each gate.  The quantum gate is applied after the linear
    transformation and before the activation function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear layers
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate modules
        self.forget_gate = QuantumGate(hidden_dim, n_qubits)
        self.input_gate = QuantumGate(hidden_dim, n_qubits)
        self.update_gate = QuantumGate(hidden_dim, n_qubits)
        self.output_gate = QuantumGate(hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)

            f = self.forget_linear(combined)
            f = self.forget_gate(f)
            f = torch.sigmoid(f)

            i = self.input_linear(combined)
            i = self.input_gate(i)
            i = torch.sigmoid(i)

            g = self.update_linear(combined)
            g = self.update_gate(g)
            g = torch.tanh(g)

            o = self.output_linear(combined)
            o = self.output_gate(o)
            o = torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)


# --------------------------------------------------------------------------- #
# Sequence tagging model using the quantum hybrid cell
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the quantum‑augmented
    :class:`HybridQLSTM`.  The rest of the architecture is identical to
    the classical version.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
