from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QLSTMRegressor(tq.QuantumModule):
    """Quantum‑enhanced LSTM regressor.

    The model replaces each classical LSTM gate with a small
    trainable quantum circuit.  A GeneralEncoder maps the
    classical gate inputs into a superposition, followed by an
    RZ rotation and a CNOT chain that entangles the wires.
    The measurement output is fed back into the standard LSTM
    recurrence, and a linear head produces a scalar regression
    target.  The architecture mirrors the classical version
    to enable direct comparison.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.rz(qdev)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_wires: int, n_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Linear projections for classical gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, num_wires)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, num_wires)
        self.update_lin = nn.Linear(input_dim + hidden_dim, num_wires)
        self.output_lin = nn.Linear(input_dim + hidden_dim, num_wires)

        # Quantum gate wrapper
        self.qgate = self.QGate(num_wires)

        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

    def _apply_qgate(self, gate_input: torch.Tensor) -> torch.Tensor:
        batch = gate_input.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=batch, device=gate_input.device)
        self.qgate.encoder(qdev, gate_input)
        return self.qgate(qdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = x.shape
        hx = torch.zeros(batch, self.hidden_dim, device=x.device)
        cx = torch.zeros(batch, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            xt = x[:, t, :]
            combined = torch.cat([xt, hx], dim=1)

            f_in = torch.sigmoid(self.forget_lin(combined))
            i_in = torch.sigmoid(self.input_lin(combined))
            g_in = torch.tanh(self.update_lin(combined))
            o_in = torch.sigmoid(self.output_lin(combined))

            f = torch.sigmoid(self._apply_qgate(f_in))
            i = torch.sigmoid(self._apply_qgate(i_in))
            g = torch.tanh(self._apply_qgate(g_in))
            o = torch.sigmoid(self._apply_qgate(o_in))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

        return self.head(hx)

def generate_superposition_sequence(num_wires: int, seq_len: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex superposition states and regression labels for a sequence."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, seq_len, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        for t in range(seq_len):
            states[i, t] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class QuantumSequenceRegressionDataset(torch.utils.data.Dataset):
    """Dataset of quantum state sequences for regression."""
    def __init__(self, samples: int, seq_len: int, num_wires: int):
        self.samples = samples
        self.seq_len = seq_len
        self.num_wires = num_wires
        self.states, self.labels = generate_superposition_sequence(num_wires, seq_len, samples)
        # Convert complex states to real features by taking the real part
        # of the first ``num_wires`` amplitudes.
        self.features = np.real(self.states[:, :, :num_wires])

    def __len__(self) -> int:  # type: ignore[override]
        return self.samples

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "sequence": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["QLSTMRegressor", "generate_superposition_sequence", "QuantumSequenceRegressionDataset"]
