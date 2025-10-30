"""Hybrid quantum‑classical regression model – quantum component only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The corresponding labels are constructed from the angles for a simple
    sinusoidal relationship.
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


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary with ``states`` and ``target``.
    The ``states`` tensor is complex and suitable for feeding into a
    :class:`torchquantum.QuantumDevice`.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QLayer(tq.QuantumModule):
    """
    Quantum gate block that encodes the input, applies trainable RX gates,
    and a CNOT chain.  The block is used for each LSTM gate.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_wires)`` containing real
            parameters for the encoder.
        Returns
        -------
        torch.Tensor
            Measurement outcomes of shape ``(batch, n_wires)``.
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)


class QuantumLSTMBlock(tq.QuantumModule):
    """
    LSTM where the four gates are implemented by small quantum circuits.
    The linear projections are classical, but the gate activations are
    produced by the quantum blocks.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = n_wires  # enforce hidden_dim == n_wires for consistency
        self.n_wires = n_wires

        # Linear projections that map the concatenated input and hidden
        # state to a vector that will be fed into the quantum gates.
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_wires)

        # Quantum gate blocks
        self.forget_gate = QLayer(n_wires)
        self.input_gate = QLayer(n_wires)
        self.update_gate = QLayer(n_wires)
        self.output_gate = QLayer(n_wires)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states
            Optional initial hidden and cell states.
        Returns
        -------
        tuple
            * outputs: ``(seq_len, batch, hidden_dim)``
            * (hx, cx): final hidden and cell states
        """
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

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class QuantumRegressionHybrid(tq.QuantumModule):
    """
    Hybrid quantum‑classical regression model that supports both
    single‑step and sequential inputs.  For single‑step data a
    parameter‑efficient encoder followed by a measurement is used.
    For sequences a quantum LSTM block is applied before the final
    linear head.
    """
    def __init__(
        self,
        num_features: int,
        num_wires: int,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires
        self.use_lstm = use_lstm

        # Encoder that maps each feature to a rotation on a separate wire.
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(num_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

        if use_lstm:
            self.quantum_lstm = QuantumLSTMBlock(
                input_dim=num_features,
                hidden_dim=num_wires,
                n_wires=num_wires,
            )
            self.lstm_head = nn.Linear(num_wires, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            ``(batch, features)`` for single‑step regression or
            ``(seq_len, batch, features)`` for sequence regression.
        Returns
        -------
        torch.Tensor
            Prediction of shape ``(batch,)`` or ``(seq_len, batch)``.
        """
        if inputs.dim() == 3:  # sequence data
            outputs, _ = self.quantum_lstm(inputs)
            preds = self.lstm_head(outputs)
            return preds.squeeze(-1)

        # single‑step data
        bsz = inputs.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=inputs.device)
        self.encoder(qdev, inputs)
        self.random_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
