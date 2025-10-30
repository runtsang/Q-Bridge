"""Hybrid quantum regression model that mirrors the classical architecture using
torchquantum variational circuits for each block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum data generation (adapted from QuantumRegression.py)
# --------------------------------------------------------------------------- #

def generate_quantum_regression_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates states of the form |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩
    and a target label y = sin(2θ) cosφ.
    """
    thetas = 2 * torch.pi * torch.rand(samples)
    phis = 2 * torch.pi * torch.rand(samples)

    # Construct basis vectors
    omega0 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega0[0] = 1.0
    omega1 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega1[-1] = 1.0

    states = torch.cos(thetas)[:, None] * omega0[None, :] + \
             torch.exp(1j * phis)[:, None] * torch.sin(thetas)[:, None] * omega1[None, :]
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels

class QuantumRegressionDataset(tq.QuantumDataset):
    """Dataset that returns quantum states and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        states, labels = generate_quantum_regression_data(num_wires, samples)
        super().__init__(states, labels)

# --------------------------------------------------------------------------- #
# Quantum building blocks
# --------------------------------------------------------------------------- #

class QCNNLayer(tq.QuantumModule):
    """A convolution‑like layer implemented with a small variational circuit."""
    def __init__(self, num_wires: int, n_ops: int = 12):
        super().__init__()
        self.num_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        return tqf.measure_all_z(qdev)

class QLSTMCell(tq.QuantumModule):
    """LSTM‑style cell where each gate is a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projections to quantum parameters
        self.linear_f = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_i = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_g = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_o = nn.Linear(input_dim + hidden_dim, n_wires)

        # Quantum gates for each gate
        self.gate_f = QCNNLayer(n_wires)
        self.gate_i = QCNNLayer(n_wires)
        self.gate_g = QCNNLayer(n_wires)
        self.gate_o = QCNNLayer(n_wires)

        # Classical post‑processing
        self.to_float = nn.Linear(n_wires, hidden_dim)

    def forward(self, qdev: tq.QuantumDevice, hx: torch.Tensor, cx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([hx, cx], dim=-1)
        f_params = self.linear_f(combined)
        i_params = self.linear_i(combined)
        g_params = self.linear_g(combined)
        o_params = self.linear_o(combined)

        # Encode parameters into quantum device
        qdev.set_params(f_params, wires=range(self.n_wires))
        f = self.gate_f(qdev)
        qdev.set_params(i_params, wires=range(self.n_wires))
        i = self.gate_i(qdev)
        qdev.set_params(g_params, wires=range(self.n_wires))
        g = self.gate_g(qdev)
        qdev.set_params(o_params, wires=range(self.n_wires))
        o = self.gate_o(qdev)

        # Classical LSTM equations
        cx_new = f * cx + i * g
        hx_new = o * torch.tanh(cx_new)
        return hx_new, cx_new

class QuantumAttention(tq.QuantumModule):
    """A minimal quantum attention block using a fixed entangling circuit."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.attn_layer = QCNNLayer(n_wires)
        self.output_layer = nn.Linear(n_wires, n_wires)

    def forward(self, qdev: tq.QuantumDevice, seq_features: torch.Tensor) -> torch.Tensor:
        """
        seq_features: (batch, seq_len, n_wires)
        """
        bsz, seq_len, _ = seq_features.shape
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=seq_features.device)
        # Encode each time step
        for t in range(seq_len):
            self.encoder(qdev, seq_features[:, t, :])
            attn = self.attn_layer(qdev)
            seq_features[:, t, :] = self.output_layer(attn)
        return seq_features

# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #

class HybridQuantumRegression(tq.QuantumModule):
    """
    Quantum analogue of HybridRegressionModel.
    Order: Feature encoder ➜ QCNN layers ➜ QLSTM ➜ QuantumAttention ➜ classical head.
    """
    def __init__(self, num_wires: int, hidden_dim: int = 32, lstm_layers: int = 1):
        super().__init__()
        self.num_wires = num_wires
        self.hidden_dim = hidden_dim

        # Feature encoding
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # QCNN front‑end
        self.conv1 = QCNNLayer(num_wires)
        self.pool1 = QCNNLayer(num_wires)
        self.conv2 = QCNNLayer(num_wires)
        self.pool2 = QCNNLayer(num_wires)
        self.conv3 = QCNNLayer(num_wires)

        # QLSTM core
        self.lstm_cells = nn.ModuleList(
            [QLSTMCell(num_wires, hidden_dim, num_wires) for _ in range(lstm_layers)]
        )

        # Quantum attention
        self.attention = QuantumAttention(num_wires)

        # Classical head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: (batch, num_wires) complex tensor representing input states.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)

        # Encode raw input
        self.encoder(qdev, state_batch)

        # QCNN sequence (treated as a single time step for simplicity)
        conv_out = self.conv1(qdev)
        conv_out = self.pool1(qdev)
        conv_out = self.conv2(qdev)
        conv_out = self.pool2(qdev)
        conv_out = self.conv3(qdev)

        # Prepare sequence tensor for LSTM (batch, seq_len=1, hidden_dim)
        seq = conv_out.unsqueeze(1)

        # LSTM forward
        hx = torch.zeros(bsz, self.hidden_dim, device=state_batch.device)
        cx = torch.zeros(bsz, self.hidden_dim, device=state_batch.device)
        for cell in self.lstm_cells:
            hx, cx = cell(qdev, hx, cx)

        # Quantum attention applied to the hidden representation
        seq = self.attention(qdev, seq)

        # Pool over sequence dimension (mean) and head
        pooled = seq.mean(dim=1)
        out = self.head(pooled).squeeze(-1)
        return out

__all__ = ["QuantumRegressionDataset", "HybridQuantumRegression", "generate_quantum_regression_data"]
