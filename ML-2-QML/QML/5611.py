"""Hybrid quantum‑classical regression model that fuses LSTM, transformer, kernel, and quantum features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

# Data generation utilities
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# Quantum feature extractor
class QuantumFeatureLayer(tq.QuantumModule):
    """Random quantum layer producing features via measurement."""
    def __init__(self, n_wires: int, n_features: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.n_features = n_features

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        return self.measure(qdev)[:, :self.n_features]

# Quantum kernel
class QuantumKernel(tq.QuantumModule):
    """Quantum kernel using a fixed ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.qdev, x, y)
        return torch.abs(self.qdev.states.view(-1)[0])

# Quantum LSTM
class QLSTM(tq.QuantumModule):
    """Quantum‑enhanced LSTM cell."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output_gate = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Quantum transformer block
class QTransformerBlock(tq.QuantumModule):
    """Quantum‑enhanced transformer block."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_qubits = n_qubits
        self.heads = nn.ModuleList([self.QLayer(n_qubits) for _ in range(num_heads)])
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        head_outputs = []
        for head in self.heads:
            flat = x.view(batch_size * seq_len, embed_dim)
            out = head(flat)
            head_outputs.append(out)
        out = torch.stack(head_outputs, dim=1).view(batch_size, seq_len, self.num_heads, embed_dim)
        out = out.mean(dim=2)  # average over heads
        return self.combine(out)

# Hybrid regression model
class QModel(nn.Module):
    """Unified regression model that can employ classical or quantum sub‑modules."""
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        use_lstm: bool = False,
        use_transformer: bool = False,
        use_kernel: bool = False,
        use_quantum: bool = False,
        kernel_dim: int = 64,
        quantum_dim: int = 64,
        n_qubits: int = 4,
    ):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer
        self.use_kernel = use_kernel
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits

        if use_lstm:
            self.lstm = QLSTM(num_features, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = None

        if use_transformer:
            self.transformer = QTransformerBlock(num_features, num_heads=4, n_qubits=num_features)
        else:
            self.transformer = None

        if use_kernel:
            self.kernel_layer = QuantumKernel(n_wires=n_qubits)
        else:
            self.kernel_layer = None

        if use_quantum:
            self.quantum_layer = QuantumFeatureLayer(n_wires=n_qubits, n_features=quantum_dim)
        else:
            self.quantum_layer = None

        # Determine input dimension for head
        input_dim = num_features
        if use_lstm:
            input_dim = hidden_dim
        if use_transformer:
            input_dim = num_features
        if use_kernel:
            input_dim += kernel_dim
        if use_quantum:
            input_dim += quantum_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lstm:
            out, _ = self.lstm(x)
            x = out[:, -1, :]
        if self.use_transformer:
            x = self.transformer(x)
            x = x.mean(dim=1)
        features = [x]
        if self.use_kernel:
            # Use a fixed basis vector for kernel evaluation
            basis = torch.randn_like(x)
            k = self.kernel_layer(x, basis)
            features.append(k.unsqueeze(-1))
        if self.use_quantum:
            qf = self.quantum_layer(x)
            features.append(qf)
        x = torch.cat(features, dim=-1)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
