import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_quantum_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    zero_state = np.zeros(2 ** num_wires, dtype=complex)
    zero_state[0] = 1.0
    one_state = np.zeros(2 ** num_wires, dtype=complex)
    one_state[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.empty((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * zero_state + np.exp(1j * phis[i]) * np.sin(thetas[i]) * one_state

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuantumRegression(tq.QuantumModule):
    class EntangleBlock(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 1):
            super().__init__()
            self.num_wires = num_wires
            self.depth = depth
            self.cz = tq.CZ(has_params=False)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.depth):
                for wire in range(self.num_wires - 1):
                    self.cz(qdev, wires=[wire, wire + 1])
                for wire in range(self.num_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, entangle_depth: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.entangle = self.EntangleBlock(num_wires, depth=entangle_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.entangle(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridQuantumRegression", "RegressionDataset", "generate_quantum_superposition_data"]
