"""Quantum regression model with a variational ansatz and entanglement."""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that generates superposition states with added noise."""

    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.05):
        self.states, self.labels = self._generate_data(num_wires, samples, noise_std)

    def _generate_data(self, num_wires, samples, noise_std):
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            omega0 = np.zeros(2 ** num_wires, dtype=complex)
            omega0[0] = 1.0
            omega1 = np.zeros(2 ** num_wires, dtype=complex)
            omega1[-1] = 1.0
            states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
        labels = np.sin(2 * thetas) * np.cos(phis) + np.random.normal(0, noise_std, size=(samples,))
        return states, labels.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Variational quantum circuit with entanglement and a classical head."""

    class VariationalAnsatz(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 3):
            super().__init__()
            self.num_wires = num_wires
            self.depth = depth
            self.params = nn.Parameter(torch.randn(depth, num_wires, 3))
            self.entanglement = tq.CNOT

        def forward(self, qdev: tq.QuantumDevice):
            for d in range(self.depth):
                for w in range(self.num_wires):
                    tq.RX(self.params[d, w, 0])(qdev, w)
                    tq.RY(self.params[d, w, 1])(qdev, w)
                    tq.RZ(self.params[d, w, 2])(qdev, w)
                # Entangle adjacent qubits
                for w in range(self.num_wires - 1):
                    self.entanglement(qdev, wires=[w, w + 1])

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.ansatz = self.VariationalAnsatz(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.ansatz(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset"]
