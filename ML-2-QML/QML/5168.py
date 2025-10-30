"""Quantum implementations of the hybrid LSTM, sampler and classifier.

Each component is a subclass of ``torchquantum.QuantumModule`` and
implements the same public interface as its classical counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class HybridQLSTM(tq.QuantumModule):
    """
    Quantum LSTM cell where each gate is a small variational
    quantum circuit.  The architecture follows the QLSTM seed but
    augments the gate rotations with a trainable CNOT ladder.
    """
    class QGate(tq.QuantumModule):
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
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QClassifierQuantum(tq.QuantumModule):
    """
    Variational quantum classifier that maps a hidden state to a two‑class
    probability distribution.  The circuit is a simple layer‑wise ansatz
    followed by measurement of the Z‑observable on each qubit.
    """
    def __init__(self, hidden_dim: int, depth: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.n_qubits = hidden_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_qubits)]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(self.n_qubits * depth)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for idx, param in enumerate(self.params):
            wire = idx % self.n_qubits
            tq.RY(param, wires=wire)(qdev)
            if idx < len(self.params) - 1:
                tgt = (wire + 1) % self.n_qubits
                tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)

class SamplerQNNQuantum(tq.QuantumModule):
    """
    Quantum sampler that produces a probability distribution over two
    outcomes from a 2‑qubit circuit.  The circuit is identical to the
    one used in the SamplerQNN reference pair.
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(4)]
        )
        self.cnot = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=2, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.cnot(qdev, wires=[0, 1])
        for idx, param in enumerate(self.params[:2]):
            tq.RY(param, wires=idx)(qdev)
        self.cnot(qdev, wires=[0, 1])
        for idx, param in enumerate(self.params[2:]):
            tq.RY(param, wires=idx)(qdev)
        return torch.softmax(self.measure(qdev), dim=-1)

__all__ = ["HybridQLSTM", "QClassifierQuantum", "SamplerQNNQuantum"]
