"""Quantum‑enhanced LSTM with a regression head.

The implementation builds on the original QML seed by adding a
quantum regression module that operates on the hidden state.  The
regression head uses a variational circuit and a linear read‑out,
illustrating how the same quantum cell can serve both sequence
tagging and regression tasks.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# Quantum utilities
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    cos(theta)|0...0> + exp(i*phi) sin(theta)|1...1>.
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
    return states, labels

class RegressionDataset(Dataset):
    """
    Dataset that returns quantum states and their regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Core quantum LSTM cell
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Quantum LSTM cell where the gates are realised by small variational circuits.
    An additional regression head operates on the hidden state using a
    quantum variational circuit followed by a linear read‑out.
    """
    class QLayer(tq.QuantumModule):
        """
        Generic quantum layer that encodes classical input and applies
        a trainable variational circuit.
        """
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
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections to the quantum space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Classical read‑outs
        self.hidden2tag = nn.Linear(hidden_dim, 1)  # placeholder for tagging head
        # Regression head
        self.hidden2reg = nn.Linear(hidden_dim, 1)

        # Mapping from hidden to quantum wires for regression
        self.hidden_to_qubits = nn.Linear(hidden_dim, n_qubits)

        # Regression quantum layer
        self.regression_layer = self.QLayer(n_qubits)
        self.regression_head = nn.Linear(n_qubits, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

        # Quantum regression on the hidden state
        q_input = self.hidden_to_qubits(hx)
        q_features = self.regression_layer(q_input)
        reg_output = self.regression_head(q_features).squeeze(-1)

        return stacked, (hx, cx), reg_output

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

# ----------------------------------------------------------------------
# Tagging / regression model
# ----------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    The quantum version also exposes a regression output per time step.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        mode: str = "tagging",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden2reg = nn.Linear(hidden_dim, 1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, QLSTM):
            lstm_out, (_, _), reg_out = self.lstm(embeds.view(len(sentence), 1, -1))
        else:
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            reg_out = None
        if self.mode == "tagging":
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return F.log_softmax(tag_logits, dim=1)
        elif self.mode == "regression":
            reg_logits = self.hidden2reg(lstm_out.view(len(sentence), -1))
            return reg_logits.squeeze(-1)
        else:  # both
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            reg_logits = self.hidden2reg(lstm_out.view(len(sentence), -1))
            return F.log_softmax(tag_logits, dim=1), reg_logits.squeeze(-1)

class QModel(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical baseline.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QLSTM", "LSTMTagger", "RegressionDataset", "QModel", "generate_superposition_data"]
