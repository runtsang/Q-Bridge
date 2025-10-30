import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Optional Gaussian noise can be added to the labels.
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
    if noise_std > 0.0:
        labels += np.random.normal(0.0, noise_std, size=labels.shape).astype(np.float32)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise_std)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class QuantumRegressionGen419(tq.QuantumModule):
    """
    A variational quantum circuit with a classical head for regression.
    The circuit includes multiple entanglement layers and a learnable mixer.
    """
    class MixerLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.entangle = tq.CNOT(has_params=False, trainable=False)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            # Apply pairwise entanglement in a ring topology
            for i in range(self.num_wires):
                self.entangle(qdev, wires=[i, (i + 1) % self.num_wires])
            # Apply parameterized rotations
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.mixer_layers = nn.ModuleList([self.MixerLayer(num_wires) for _ in range(depth)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        batch_size = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=batch_size, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for mixer in self.mixer_layers:
            mixer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state_batch)

def get_optimizer_and_scheduler(
    model: tq.QuantumModule,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    scheduler_step: int = 10,
    scheduler_gamma: float = 0.5
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    return optimizer, scheduler

__all__ = [
    "QuantumRegressionGen419",
    "RegressionDataset",
    "generate_superposition_data",
    "get_optimizer_and_scheduler"
]
