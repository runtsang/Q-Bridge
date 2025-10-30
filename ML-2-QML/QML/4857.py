from __future__ import annotations

import numpy as np
from typing import Sequence
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict

class QuantumGateLayer(tq.QuantumModule):
    """
    Small quantum gate block that can be reused for each LSTM gate.

    The circuit consists of an input encoder followed by a trainable RX rotation
    on each wire and a chain of CNOTs that entangles the wires.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        for i in range(self.n_wires - 1):
            tqf.cnot(q_device, wires=[i, i + 1])
        tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
        return self.measure(q_device)

class QuantumLSTM(nn.Module):
    """
    LSTM cell where each gate is realised by a small quantum circuit.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QuantumGateLayer(n_qubits)
        self.input_gate = QuantumGateLayer(n_qubits)
        self.update_gate = QuantumGateLayer(n_qubits)
        self.output_gate = QuantumGateLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outs.append(hx.unsqueeze(0))
        out = torch.cat(outs, dim=0)
        return out, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel based on a parameter‑shifted Ry encoder.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        self.ansatz(q_device, x)
        self.ansatz(q_device, -y)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute Gram matrix using the quantum kernel.
    """
    kernel = QuantumKernel()
    return np.array([[kernel.kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumGateLayer", "QuantumLSTM", "QuantumKernel", "kernel_matrix"]
