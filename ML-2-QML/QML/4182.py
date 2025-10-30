"""Quantum‑enhanced LSTM with a regression head.

The class :class:`HybridQLSTM` mirrors the classical implementation
but replaces each gate with a small variational circuit using
torchquantum.  The regression head is identical to the classical
variant, making the two modules API‑compatible.  A helper dataset
produces superposition states and labels, matching the quantum
regression example.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset


class HybridQLSTM(nn.Module):
    """Quantum LSTM cell with per‑time‑step regression output."""
    class QLayer(tq.QuantumModule):
        """Small variational circuit used for each LSTM gate."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple encoding of the linear projection into qubit angles
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            # Trainable single‑qubit rotations
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.output_dim = output_dim

        # Quantum gates for each LSTM component
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections into qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Regression head
        self.regressor = nn.Linear(n_qubits, output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        regress = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))
            regress.append(self.regressor(hx).unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        regress_out = torch.cat(regress, dim=0)
        return stacked, regress_out, (hx, cx)

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


class RegressionDataset(Dataset):
    """Quantum‑style dataset generating superposition states and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate(samples, num_wires)

    @staticmethod
    def _generate(samples: int, num_wires: int) -> tuple[np.ndarray, np.ndarray]:
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

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["HybridQLSTM", "RegressionDataset"]
