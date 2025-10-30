"""Hybrid quantum self‑attention with quantum kernel and regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

# --- Data utilities (from QuantumRegression seed) ---
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --- Quantum kernel ansatz ---
class KernalAnsatz(tq.QuantumModule):
    """
    Encodes two classical vectors into a quantum state and returns the
    absolute overlap, i.e. a quantum‑inspired RBF kernel.
    """
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """
    Quantum kernel evaluation using a fixed Ry‑based ansatz.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# --- Hybrid Self‑Attention (quantum) ---
class HybridSelfAttention(tq.QuantumModule):
    """
    Quantum self‑attention that uses a quantum kernel to calculate attention
    scores, a quantum encoder to derive value representations, and a
    classical linear head for regression.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.kernel = Kernel(n_wires)
        self.value_mapper = nn.Linear(n_wires, n_wires, bias=False)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, n_wires) where each row is a
            classical vector to be encoded.

        Returns
        -------
        torch.Tensor
            Scalar prediction for each sample: shape (batch,).
        """
        batch = inputs.shape[0]
        # Encode each input into a quantum state and measure Pauli‑Z expectation
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=inputs.device)
        self.encoder(qdev, inputs)
        # Value vectors from measurement
        values = tq.MeasureAll(tq.PauliZ)(qdev).to(torch.float32)  # (B, n_wires)

        # Compute pairwise quantum kernel matrix
        kernel_vals = torch.empty((batch, batch), device=inputs.device)
        for i in range(batch):
            for j in range(batch):
                kernel_vals[i, j] = self.kernel(inputs[i].unsqueeze(0), inputs[j].unsqueeze(0))

        # Attention weights via softmax over kernel similarities
        weights = torch.softmax(kernel_vals, dim=-1)  # (B, B)

        # Weighted sum of value vectors
        attn_out = weights @ values  # (B, n_wires)

        # Map to embedding space and regress
        mapped = self.value_mapper(attn_out)
        return self.head(mapped).squeeze(-1)

__all__ = ["HybridSelfAttention", "RegressionDataset", "generate_superposition_data"]
