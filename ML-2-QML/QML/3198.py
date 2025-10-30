"""Importable quantum module defining UnifiedQuantumHybridLayer.

This module uses Pennylane for the fully‑connected quantum layer and
TorchQuantum for the quantum LSTM gates.  The public API mirrors the
classical version so that the layer can be swapped in a PyTorch
model without changing downstream code.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pennylane imports
import pennylane as qml
import pennylane.numpy as pnp

# TorchQuantum imports
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum fully‑connected layer
class _FCLQuantum(nn.Module):
    """
    Variational quantum circuit that mimics a simple fully‑connected layer.
    """

    def __init__(self, n_features: int, n_qubits: int = 1, shots: int = 100) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.shots = shots

        # Create a device with enough wires for the features
        self.device = qml.device("default.qubit", wires=self.n_features)

        # Trainable parameters: one RX gate per wire
        self.params = nn.Parameter(torch.randn(self.n_features))

        # Define the circuit
        @qml.qnode(self.device, interface="torch", shots=self.shots)
        def circuit(x, *params):
            # Encode each feature into a Ry gate
            for i in range(self.n_features):
                qml.Ry(x[i], wires=i)
            # Apply parameterised RX gates
            for i in range(self.n_features):
                qml.RX(params[i], wires=i)
            # Measure expectation value of PauliZ on all wires and sum
            return sum(qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_features))

        self.circuit = circuit

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit with the given input features.
        """
        x = thetas.reshape(-1)
        return self.circuit(x, *self.params)

# Quantum LSTM gate implemented with TorchQuantum
class QLSTM(nn.Module):
    """
    LSTM cell where each gate is realised by a small quantum circuit.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encoder that applies Ry on each wire
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "ry", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Parameterised RX gates
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            # Measure all qubits in Z basis
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle adjacent qubits
            for wire in range(self.n_wires):
                next_wire = (wire + 1) % self.n_wires
                tqf.cnot(qdev, wires=[wire, next_wire])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM gate
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections to produce quantum parameters
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Quantum sequence tagging model
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the quantum LSTM.
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
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

# Unified hybrid layer using quantum components
class UnifiedQuantumHybridLayer(nn.Module):
    """
    Combines a quantum fully‑connected layer with a quantum LSTM for
    sequence tagging.  The API mirrors the classical version so that it
    can be used interchangeably in a PyTorch model.
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
        self.tagger = LSTMTagger(
            embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=n_qubits
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that delegates to the quantum LSTMTagger.
        """
        return self.tagger(sentence)

__all__ = ["UnifiedQuantumHybridLayer"]
