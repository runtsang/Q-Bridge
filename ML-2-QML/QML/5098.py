"""
Quantum‑enhanced hybrid estimator that mirrors :class:`ml.EstimatorQNNGen490`.

The implementation replaces the linear gates with small variational quantum
circuits.  It can be used as a plug‑in for the classical version by simply
importing this module instead of the classical one.

The public API is intentionally identical to the classical version so that
experiments can be run on either backend with minimal code changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torch.quantum import QuantumModule, QuantumDevice, MeasureAll, PauliZ
from torch.quantum.functional import cnot
from torch.quantum import GeneralEncoder, RandomLayer, RX, RY
import numpy as np
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data in superposition form
    (cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>).
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
    Dataset yielding quantum states and corresponding regression targets.
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


class QLSTMQuantum(QuantumModule):
    """
    LSTM cell where gates are realised by small variational quantum circuits.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = MeasureAll(PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    cnot(qdev, wires=[wire, 0])
                else:
                    cnot(qdev, wires=[wire, wire + 1])
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QuanvolutionFilterQuantum(tq.QuantumModule):
    """
    Apply a random two‑qubit quantum kernel to 2×2 image patches.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class RegressionQuantumHead(tq.QuantumModule):
    """
    Simple quantum head that maps the last layer of the quantum device to a scalar.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = RX(has_params=True, trainable=True)
            self.ry = RY(has_params=True, trainable=True)

        def forward(self, qdev: QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = GeneralEncoder(GeneralEncoder.get_encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = MeasureAll(PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = QuantumDevice(self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


class EstimatorQNNGen490(tq.QuantumModule):
    """
    Full quantum‑enhanced estimator.  The API matches the classical
    :class:`ml.EstimatorQNNGen490` so that experiments can be run on either
    backend by swapping imports.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        n_qubits: int = 0,
        use_lstm: bool = False,
        use_quanvolution: bool = False,
        use_quantum_head: bool = True,
        regression_wires: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        self.use_quanvolution = use_quanvolution
        self.use_quantum_head = use_quantum_head

        # Base feed‑forward block (classical, but wrapped in QuantumModule for API)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Optional LSTM
        if self.use_lstm:
            if self.n_qubits > 0:
                self.lstm = QLSTMQuantum(input_dim, hidden_dim, n_qubits)
            else:
                self.lstm = None  # will be handled in forward
        else:
            self.lstm = None

        # Optional qua‑convolution
        if self.use_quanvolution:
            self.qfilter = QuanvolutionFilterQuantum()
            self.conv_out_dim = 4 * 14 * 14
        else:
            self.qfilter = None
            self.conv_out_dim = input_dim

        # Regression head
        if self.use_quantum_head:
            self.head = RegressionQuantumHead(regression_wires)
        else:
            self.head = nn.Linear(self.conv_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            If ``use_quanvolution`` is ``True`` the tensor must have shape
            ``(batch, 1, 28, 28)``.  Otherwise it should be
            ``(batch, seq_len, features)`` if ``use_lstm`` is ``True`` or
            ``(batch, features)`` otherwise.
        """
        # 1. Qua‑convolution / classical conv
        if self.qfilter is not None:
            features = self.qfilter(x)
        else:
            if self.lstm is not None and x.dim() == 3:
                # Sequence data for quantum LSTM
                features, _ = self.lstm(x)
                features = features[:, -1, :]  # last hidden state
            else:
                features = x

        # 2. Feed‑forward
        ff_out = self.ff(features)

        # 3. Head
        out = self.head(ff_out)

        return out.squeeze(-1)


__all__ = [
    "EstimatorQNNGen490",
    "RegressionDataset",
    "generate_superposition_data",
    "QLSTMQuantum",
    "QuanvolutionFilterQuantum",
    "RegressionQuantumHead",
]
