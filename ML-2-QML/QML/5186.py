"""Hybrid LSTM with quantum gates and QCNN feature extractor.

The module mirrors the classical version but replaces the LSTM gates
with parametrised quantum circuits (via TorchQuantum) and uses a
quantum SamplerQNN to modulate the gates.  The QCNN feature extractor
is a shallow quantum circuit that emulates the classical QCNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum import encoder_op_list_name_dict
from torch.utils.data import Dataset
import numpy as np

__all__ = [
    "HybridQLSTM",
    "QuantumHybridLSTM",
    "SamplerQNN",
    "QCNNQuantumModel",
    "RegressionDataset",
    "QModel",
]


class QCNNQuantumModel(tq.QuantumModule):
    """Quantum analogue of the classical QCNN feature extractor.

    The circuit follows the same layering pattern (convolution + pooling)
    but all operations are implemented with TorchQuantum gates.  The
    observable is a Pauli‑Z measurement on each qubit, producing a
    feature vector of length equal to the number of qubits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_qubits = 8
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_qubits)
            ]
        )
        self.conv = tq.RX(has_params=True, trainable=True)
        self.pool = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, qdev.input)
        # Convolutional layer
        for wire in range(self.n_qubits // 2):
            self.conv(qdev, wires=[wire, wire + 1])
        # Pooling layer
        for wire in range(0, self.n_qubits, 2):
            self.pool(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class SamplerQNN(tq.QuantumModule):
    """Quantum sampler that outputs a probability vector used to modulate gates."""

    def __init__(self) -> None:
        super().__init__()
        self.n_qubits = 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.cnot = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, qdev.input)
        self.cnot(qdev, wires=[0, 1])
        return self.measure(qdev).abs()  # probabilities


class QuantumHybridLSTM(tq.QuantumModule):
    """LSTM core where each gate is a small quantum circuit."""

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, qdev.input)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev).abs()

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.sampler = SamplerQNN()

    def _init_states(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def _modulate(self, vec: torch.Tensor) -> torch.Tensor:
        """Use the sampler to produce a scalar modulation factor per sample."""
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=vec.size(0), device=vec.device)
        qdev.input = vec
        out = self.sampler(qdev)
        return out.mean(dim=1, keepdim=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hx, cx = self._init_states(inputs)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_in = self.linear_forget(combined)
            i_in = self.linear_input(combined)
            g_in = self.linear_update(combined)
            o_in = self.linear_output(combined)
            f = torch.sigmoid(self.forget(f_in))
            i = torch.sigmoid(self.input(i_in))
            g = torch.tanh(self.update(g_in))
            o = torch.sigmoid(self.output(o_in))
            mod = self._modulate(f_in)
            f = f * mod
            i = i * mod
            o = o * mod
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


class HybridQLSTM(tq.QuantumModule):
    """Quantum‑enhanced tagger that combines QCNN, quantum LSTM and a linear head."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.qcnn = QCNNQuantumModel()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Use the QCNN output dimension as LSTM input dimension
        self.lstm = QuantumHybridLSTM(self.qcnn.n_qubits, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # QCNN expects a quantum device; we use a dummy input
        qdev = tq.QuantumDevice(n_wires=self.qcnn.n_qubits, bsz=embeds.size(0), device=embeds.device)
        qdev.input = embeds
        qcnn_out = self.qcnn(qdev)
        lstm_out, _ = self.lstm(qcnn_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)


class RegressionDataset(Dataset):
    """Quantum regression dataset with superposition states."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate(samples, num_wires)

    @staticmethod
    def _generate(samples: int, num_wires: int):
        omega0 = np.zeros(2**num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2**num_wires, dtype=complex)
        omega1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2**num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Quantum regression head that measures qubits and applies a linear head."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            return tq.MeasureAll(tq.PauliZ)(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)
