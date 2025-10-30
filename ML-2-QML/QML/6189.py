"""Quantum‑enhanced LSTM with interchangeable gate circuits.

The QLSTM class supports two styles of quantum gates:
* SimpleQGate – a lightweight 1‑qubit parameterised circuit.
* ComplexQLayer – a richer circuit with a general encoder and a
  CNOT‑based entangling pattern.
Both gates are wrapped in torchquantum.QuantumModule, allowing
gradient‑based optimisation with PyTorch.
"""

import torch
import torch.nn as nn
import torch.quantum as tq  # PyTorch Quantum
import torch.quantum.functional as tqf
from typing import Tuple

class SimpleQGate(tq.QuantumModule):
    """1‑qubit parameterised gate using a single RX rotation."""
    def __init__(self, n_wires: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder: map classical inputs to qubit rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "rx", "wires": [0]}]
        )
        # Trainable RX gate
        self.param_gate = tq.RX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        self.param_gate(qdev, wires=0)
        return self.measure(qdev)

class ComplexQLayer(tq.QuantumModule):
    """Rich quantum gate with general encoder and entanglement."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """Quantum LSTM where each gate is a quantum module."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 simple_gate: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_cls = SimpleQGate if simple_gate else ComplexQLayer
        self.forget = gate_cls(n_qubits)
        self.input_g = gate_cls(n_qubits)
        self.update = gate_cls(n_qubits)
        self.output = gate_cls(n_qubits)

        # Linear projections into qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_g(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 4,
                 simple_gate: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim,
                          hidden_dim,
                          n_qubits=n_qubits,
                          simple_gate=simple_gate)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.nn.functional.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
