"""Quantum‑enhanced LSTM with tunable entanglement depth and hybrid loss logging."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import pennylane as qml

class QLSTM(nn.Module):
    """A quantum LSTM that replaces each gate with a variational circuit."""

    class _QuantumGate(nn.Module):
        """Variational circuit for one gate. Depth is controlled by `depth`."""

        def __init__(self, n_wires: int, depth: int = 1) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Parameter matrix: depth × n_wires
            self.params = nn.Parameter(torch.randn(depth, n_wires))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the variational circuit to input angles `x`."""
            @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface="torch")
            def circuit(inputs, params):
                # Encode input as rotation angles on each qubit
                for w in range(self.n_wires):
                    qml.RX(inputs[:, w], wires=w)
                # Variational layers
                for d in range(self.depth):
                    for w in range(self.n_wires):
                        qml.RX(params[d, w], wires=w)
                    # Entanglement layer
                    for w in range(self.n_wires - 1):
                        qml.CNOT(wires=[w, w + 1])
                # Return expectation values of Pauli‑Z
                return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

            return circuit(x, self.params)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = n_qubits  # quantum gates output n_qubits, set hidden_dim to match
        self.n_qubits = n_qubits
        self.depth = depth

        # Quantum gates for each LSTM component
        self.forget = self._QuantumGate(n_qubits, depth)
        self.input = self._QuantumGate(n_qubits, depth)
        self.update = self._QuantumGate(n_qubits, depth)
        self.output = self._QuantumGate(n_qubits, depth)

        # Linear maps to feed into the quantum circuits
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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

    def log_quantum_metrics(self) -> dict:
        """Return a dictionary of useful quantum training metrics."""
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if n_qubits > 0:
            self.hidden_dim = n_qubits
            self.lstm = QLSTM(embedding_dim, self.hidden_dim, n_qubits=n_qubits, depth=depth)
        else:
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(self.hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
