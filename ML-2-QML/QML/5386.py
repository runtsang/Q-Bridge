"""Hybrid quantum‑classical binary classifier.

This module implements the same architecture as the classical variant but
replaces the kernel surrogate and the classification head with
parameterised quantum circuits.  A TorchQuantum kernel is used to
embed the CNN features, and a small two‑qubit circuit provides the
probability estimate.  An optional QLSTM branch demonstrates how
sequential data can be processed quantum‑ly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QKernelFeatureMap(tq.QuantumModule):
    """Quantum kernel feature map using a fixed ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x)
        self.ansatz(self.q_device, -y)
        self.q_device.measure_all()
        return torch.abs(self.q_device.states[:, 0]).mean(dim=0, keepdim=True)


class QExpectationHead(tq.QuantumModule):
    """Two‑qubit expectation head."""
    def __init__(self, n_qubits: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder(self.q_device, x)
        for idx in range(self.n_qubits):
            tqf.rx(self.q_device, wires=idx, params=self.params[idx])
        self.measure(self.q_device)
        return self.q_device.states[:, 0].real.mean(dim=0, keepdim=True)


class QLSTMBranch(tq.QuantumModule):
    """Quantum LSTM branch inspired by the QLSTM implementation."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.q_device = tq.QuantumDevice(n_wires=n_wires)
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.encoder(self.q_device, x)
            for idx, gate in enumerate(self.params):
                gate(self.q_device, wires=idx)
            for idx in range(self.n_wires - 1):
                tqf.cnot(self.q_device, wires=[idx, idx + 1])
            tqf.cnot(self.q_device, wires=[self.n_wires - 1, 0])
            self.measure(self.q_device)
            return self.q_device.states[:, 0].real

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        hx = torch.zeros(batch_size, self.n_qubits, device=inputs.device)
        cx = torch.zeros(batch_size, self.n_qubits, device=inputs.device)
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
        return torch.cat(outputs, dim=0)


class HybridBinaryClassifier(tq.QuantumModule):
    """Quantum‑classical hybrid classifier with CNN backbone,
    quantum kernel feature map and quantum expectation head.
    An optional QLSTM branch can be enabled for sequential data.
    """
    def __init__(
        self,
        use_kernel: bool = True,
        kernel_wires: int = 4,
        n_qubits_head: int = 2,
        use_qlstm: bool = False,
        qlstm_params: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        dummy = torch.zeros(1, 3, 32, 32)
        flat_dim = self.backbone(dummy).view(1, -1).size(1)
        self.flatten = nn.Flatten()
        self.kernel = QKernelFeatureMap(kernel_wires) if use_kernel else None
        head_input = kernel_wires if use_kernel else flat_dim
        self.head = QExpectationHead(n_qubits_head)
        self.use_qlstm = use_qlstm
        if use_qlstm and qlstm_params is not None:
            inp_dim, hid_dim, n_q = qlstm_params
            self.qlstm = QLSTMBranch(inp_dim, hid_dim, n_q)
        else:
            self.qlstm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        if self.kernel is not None:
            x = self.kernel(x, x).squeeze(-1)
        x = x.unsqueeze(-1)
        out = self.head(x)
        probs = torch.sigmoid(out)
        if self.qlstm is not None:
            seq_out = self.qlstm(x.squeeze(-1))
            probs = torch.cat((probs, seq_out), dim=-1)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier"]
