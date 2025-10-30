"""Quantum regression model and supporting layers.

The implementation merges the concepts from the original quantum
regression example, the quantum LSTM and the quanvolution filter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumConvFilter(tq.QuantumModule):
    """Quantum implementation of a simple 2×2 filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        # each input value is mapped to an RX rotation angle
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_qubits)]
        )
        self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, kernel_size, kernel_size)"""
        bsz = x.shape[0]
        x_flat = x.view(bsz, -1)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x_flat)
        self.random_layer(qdev)
        features = self.measure(qdev)  # (batch, n_qubits)
        # average over qubits to obtain a scalar per sample
        return features.mean(dim=1, keepdim=True)


class QLSTM(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a small quantum circuit."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

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
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridRegression(tq.QuantumModule):
    """Quantum regression model that can operate in several modes."""
    def __init__(self, num_features: int, n_qubits: int, mode: str = "regression"):
        super().__init__()
        if mode not in {"regression", "sequence", "conv"}:
            raise ValueError(f"Unsupported mode {mode!r}")
        self.mode = mode
        if mode == "regression":
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(num_features)
                ]
            )
            self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_features)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.head = nn.Linear(num_features, 1)
        elif mode == "sequence":
            self.lstm = QLSTM(num_features, hidden_dim=32, n_qubits=n_qubits)
            self.head = nn.Linear(n_qubits, 1)
        else:  # conv
            self.conv = QuantumConvFilter(kernel_size=int(np.sqrt(num_features)))
            self.head = nn.Linear(1, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.mode == "regression":
            bsz = batch.shape[0]
            qdev = tq.QuantumDevice(n_wires=batch.shape[1], bsz=bsz, device=batch.device)
            self.encoder(qdev, batch)
            self.q_layer(qdev)
            features = self.measure(qdev)
            return self.head(features).squeeze(-1)
        elif self.mode == "sequence":
            outputs, _ = self.lstm(batch)
            # take the last hidden state
            last = outputs[-1]
            return self.head(last).squeeze(-1)
        else:  # conv
            features = self.conv(batch)
            return self.head(features).squeeze(-1)


__all__ = [
    "HybridRegression",
    "RegressionDataset",
    "generate_superposition_data",
    "QLSTM",
    "QuantumConvFilter",
]
