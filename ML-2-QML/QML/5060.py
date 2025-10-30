import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice, GeneralEncoder, RandomLayer, MeasureAll, PauliZ

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset for a quantum device."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis   = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegression(tq.QuantumModule):
    """Quantum regression head leveraging a variational ansatz."""
    def __init__(self, num_qubits: int, depth: int = 2):
        super().__init__()
        self.n_wires = num_qubits
        # Encoder: learnable RX rotations parameterised by classical data
        self.encoder = GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_qubits}xRy"])
        # Variational block: depth‑controlled random layer
        self.q_layer = RandomLayer(n_ops=depth * num_qubits, wires=list(range(num_qubits)))
        self.measure = MeasureAll(PauliZ)
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical inputs into rotation angles
        self.encoder(qdev, state_batch)
        # Variational processing
        self.q_layer(qdev)
        # Expectation values as classical features
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
