"""
Quantum‑enhanced LSTM with depth‑controlled quantum gates.
The implementation uses TorchQuantum to build parameterised circuits
for each gate.  The public API matches the classical version while
providing additional hyper‑parameters for circuit depth and a
variational read‑out.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class _QuantumGate(nn.Module):
    """Parameterised quantum gate block used for LSTM gates."""
    def __init__(self, n_wires: int, depth: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth

        # Encode each input element to a rotation on a distinct qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(min(n_wires, 4))
            ]
        )

        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit and return a single‑qubit measurement."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for _ in range(self.depth):
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            # Entangling layer
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return tq.MeasureAll(tq.PauliZ)(qdev)


class QLSTMCellQuantum(nn.Module):
    """Quantum LSTM cell where each gate is a quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget = _QuantumGate(n_qubits, depth=depth)
        self.input = _QuantumGate(n_qubits, depth=depth)
        self.update = _QuantumGate(n_qubits, depth=depth)
        self.output = _QuantumGate(n_qubits, depth=depth)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = states
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, (hx, cx)


class QLSTMStackQuantum(nn.Module):
    """Stack of quantum LSTM cells."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [QLSTMCellQuantum(input_dim if i == 0 else hidden_dim,
                              hidden_dim, n_qubits, depth=depth)
             for i in range(num_layers)]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = inputs.size(1)
        if init_states is None:
            hx = torch.zeros(batch_size, self.cells[0].hidden_dim, device=inputs.device)
            cx = torch.zeros(batch_size, self.cells[0].hidden_dim, device=inputs.device)
        else:
            hx, cx = init_states

        outputs = []
        for t in range(inputs.size(0)):
            x = inputs[t]
            for cell in self.cells:
                x, (hx, cx) = cell(x, (hx, cx))
            outputs.append(x.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


class QLSTMQuantum(nn.Module):
    """Drop‑in quantum LSTM that mirrors the classical API."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_stack = QLSTMStackQuantum(input_dim, hidden_dim, n_qubits, depth=depth, num_layers=1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm_stack(inputs, init_states=states)


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
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMQuantum", "QLSTMCellQuantum", "QLSTMStackQuantum", "LSTMTagger"]
