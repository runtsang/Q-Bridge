"""
Quantum components for the UnifiedHybridModel.

Provides:
  * QuantumFeatureEncoder – a variational circuit acting on a 4‑dimensional vector.
  * QuantumGate – a small circuit used inside each LSTM gate.
  * HybridCNNQuantum – classical CNN backbone with a quantum encoder.
  * HybridQLSTMQuantum – hybrid LSTM that replaces each gate with a QuantumGate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumFeatureEncoder(tq.QuantumModule):
    """
    Variational encoder used in the CNN branch.
    """

    def __init__(self, n_wires: int = 4, device: str | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.device = device or "cpu"
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "rz", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "rz", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=self.device, record_op=True)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)


class QuantumGate(tq.QuantumModule):
    """
    Quantum gate used inside each LSTM gate.
    """

    def __init__(self, n_wires: int = 4, device: str | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.device = device or "cpu"
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "rz", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=self.device, record_op=True)
        self.encoder(qdev, x)
        for w, gate in enumerate(self.params):
            gate(qdev, wires=w)
        for w in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[w, w + 1], static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0], static=self.static_mode, parent_graph=self.graph)
        return self.measure(qdev)


class HybridCNNQuantum(nn.Module):
    """
    Classical CNN backbone with a quantum encoder applied to the pooled features.
    """

    def __init__(self, in_channels: int, num_classes: int, device: str | None = None):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)
        self.quantum_encoder = QuantumFeatureEncoder(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        encoded = self.quantum_encoder(pooled)
        features = self.backbone(x)
        out = self.head(features)
        return self.norm(out)


class HybridQLSTMQuantum(nn.Module):
    """
    Hybrid LSTM cell that replaces each gate with a lightweight quantum circuit.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, device: str | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = QuantumGate(n_wires=n_qubits, device=device)
        self.input_gate = QuantumGate(n_wires=n_qubits, device=device)
        self.update_gate = QuantumGate(n_wires=n_qubits, device=device)
        self.output_gate = QuantumGate(n_wires=n_qubits, device=device)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


__all__ = ["QuantumFeatureEncoder", "QuantumGate", "HybridCNNQuantum", "HybridQLSTMQuantum"]
