"""Hybrid LSTM with quantum gates and quantum regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional

class QLSTM(tq.QuantumModule):
    """Quantum-enhanced LSTM cell with optional regression head."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, qdev.state)
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: Optional[int] = None, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.n_qubits = n_qubits
        # Quantum gates for LSTM
        self.forget_gate = QLSTM.QGate(n_qubits)
        self.input_gate = QLSTM.QGate(n_qubits)
        self.update_gate = QLSTM.QGate(n_qubits)
        self.output_gate = QLSTM.QGate(n_qubits)
        # Classical linear layers to map concatenated inputs to qubit space
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        # Heads
        if tagset_size is not None:
            self.tag_head = nn.Linear(hidden_dim, tagset_size)
        self.regression_head = nn.Linear(n_qubits, 1)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def forward_tagging(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.forward(inputs)
        logits = self.tag_head(lstm_out)
        return torch.log_softmax(logits, dim=1)

    def forward_regression(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.forward(inputs)
        # Encode final hidden state into qubits for regression
        bsz = lstm_out.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=lstm_out.device)
        encoded = self.forget_linear(lstm_out)
        self.forget_gate.encoder(qdev, encoded)
        features = self.forget_gate.measure(qdev)
        return self.regression_head(features).squeeze(-1)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(tq.QuantumModule):
    """Sequence tagging model using QLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, tagset_size=tagset_size, n_qubits=n_qubits)
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        return self.lstm.forward_tagging(embeds.view(len(sentence), 1, -1))

class RegressionDataset(Dataset):
    """Quantum regression dataset."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
    def __len__(self) -> int:
        return len(self.states)
    def __getitem__(self, idx: int) -> dict:
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
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

class RegressionModel(tq.QuantumModule):
    """Quantum regression model."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QLSTM.QGate(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QLSTM", "LSTMTagger", "RegressionDataset", "RegressionModel", "generate_superposition_data"]
