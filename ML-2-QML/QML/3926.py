"""Unified hybrid regression model with quantum LSTM for time‑series."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1) Dataset helpers – identical to the Quantum seed, but with optional
#    support for batched quantum feature vectors.
# --------------------------------------------------------------------------- #
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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields a dict with quantum state tensors and targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 2) Quantum gate‑level encoder – borrowed from the QML seed.
# --------------------------------------------------------------------------- #
class QGateEncoder(tq.QuantumModule):
    """Variational circuit that encodes a 1‑D signal into a multi‑wire QDevice."""
    def __init__(self, n_wires: int, n_params: int = 30) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(
            n_ops=n_params, wires=list(range(n_wires))
        )
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Classical features of shape (batch, seq_len, n_wires)
        Returns
        -------
        torch.Tensor
            Quantum states ready for the QLSTM.
        """
        bsz, seq_len, _ = state_batch.shape
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz * seq_len, device=state_batch.device)
        # Flatten batch and seq_len for encoding
        flat = state_batch.reshape(bsz * seq_len, -1)
        self.random_layer(qdev, flat)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        return qdev


# --------------------------------------------------------------------------- #
# 3) Quantum LSTM – gate‑level quantum circuits for each gate.
# --------------------------------------------------------------------------- #
class QuantumQLSTM(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""

    class QLayer(tq.QuantumModule):
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

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

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


# --------------------------------------------------------------------------- #
# 4) Unified regression model – quantum backbone.
# --------------------------------------------------------------------------- #
class UnifiedRegressionQLSTM(tq.QuantumModule):
    """
    Hybrid regression model that combines a quantum encoder with a
    quantum‑aware LSTM for time‑series regression.  The model can be trained
    end‑to‑end on batched quantum states.
    """
    def __init__(
        self,
        num_wires: int,
        hidden_dim: int,
        output_dim: int = 1,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = QGateEncoder(num_wires)
        self.q_lstm = QuantumQLSTM(num_wires, hidden_dim, n_qubits=num_wires)
        self.head = nn.Linear(num_wires, output_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Input tensor of shape (batch, seq_len, num_wires)
        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch, seq_len, output_dim)
        """
        encoded = self.encoder(state_batch)
        lstm_out, _ = self.q_lstm(encoded)
        preds = self.head(lstm_out)
        return preds

__all__ = ["UnifiedRegressionQLSTM", "RegressionDataset", "generate_superposition_data"]
