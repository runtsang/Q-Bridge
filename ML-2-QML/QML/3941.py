"""Quantum‑enhanced regression and sequence model.

The ``QuantumHybridRegressionQLSTM`` class mirrors the classical
``HybridRegressionQLSTM`` but replaces the feed‑forward encoder and the
LSTM with quantum‑parameterised circuits.  The model is end‑to‑end
trainable and serves as a benchmarking point for quantum advantage in
sequence regression.

Author: <your name>
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset
from typing import Tuple, Optional

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    """
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
    """
    Dataset that returns a quantum state and its regression target.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionCircuit(tq.QuantumModule):
    """
    Variational regression circuit with a random layer and single‑qubit rotations.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, state_batch: torch.Tensor) -> torch.Tensor:
        self.random_layer(qdev)
        self.encoder(qdev, state_batch)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        return self.measure(qdev)

class QuantumQLSTM(tq.QuantumModule):
    """
    LSTM cell where gates are realised by small quantum circuits.
    """
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
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QuantumHybridRegressionQLSTM(tq.QuantumModule):
    """
    Quantum‑enhanced regression + LSTM model.

    The quantum regression circuit encodes each feature vector into a
    super‑position state; the quantum LSTM processes the sequence of
    encoded states; a linear head maps the final hidden state to a
    scalar output.  The model is fully trainable end‑to‑end with
    gradient back‑propagation through the quantum device.
    """
    def __init__(
        self,
        num_features: int,
        num_wires: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = QuantumRegressionCircuit(num_wires)
        self.lstm = QuantumQLSTM(input_dim=num_wires, hidden_dim=hidden_dim, n_qubits=num_wires)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, num_features)
        """
        batch, seq_len, _ = x.shape
        x_flat = x.view(batch * seq_len, -1)
        qdev = tq.QuantumDevice(n_wires=self.encoder.num_wires, bsz=batch * seq_len, device=x.device)
        encoded = self.encoder(qdev, x_flat)
        encoded = encoded.view(batch, seq_len, -1)
        encoded = encoded.permute(1, 0, 2)  # (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(encoded)
        lstm_out = lstm_out.permute(1, 0, 2)  # (batch, seq_len, hidden_dim)
        out = self.regressor(lstm_out).squeeze(-1)
        return out

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumRegressionCircuit",
    "QuantumQLSTM",
    "QuantumHybridRegressionQLSTM",
]
