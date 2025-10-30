from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

import torchquantum as tq
import torchquantum.functional as tqf

class QuantumEncoder(tq.QuantumModule):
    """Encode two real‑valued features into a 2‑qubit state via RX/RY rotations."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=2, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        return self.measure(qdev)  # shape (batch, 2)

class EstimatorQNN(nn.Module):
    """Hybrid classical‑quantum regressor.  When `use_quantum` is True the
    input is first encoded by a 2‑qubit circuit, otherwise a raw
    linear stack is used."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 use_quantum: bool = False,
                 n_qubits: int = 2) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            self.encoder = QuantumEncoder()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            quantum_features = self.encoder(inputs)
            return self.net(quantum_features)
        return self.net(inputs)

class QLSTM(nn.Module):
    """LSTM cell where each gate is a tiny quantum sub‑module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
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
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
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

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

__all__ = ["EstimatorQNN", "QLSTM"]
