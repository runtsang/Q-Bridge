import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form
    cos(theta)|0…0> + e^{i phi} sin(theta)|1…1> and a non‑linear target."""
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
    """Dataset that returns a complex state vector and a real target."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuantumRegressionModel(tq.QuantumModule):
    """Hybrid quantum regression model that mirrors the classical CNN but
    uses a variational quantum circuit for the feature extraction step.
    The architecture is a fusion of the Quantum‑NAT encoder,
    the QLayer from the QLSTM example, and the regression head
    from the original seed."""
    class QLayer(tq.QuantumModule):
        """Small variational layer that applies a random circuit followed by
        a trainable RX gate on every wire."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=range(n_wires))
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Classical feature extractor (same as the CNN in the ML version)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        dummy = torch.randn(
            1, 1, int(np.ceil(np.sqrt(num_wires))), int(np.ceil(np.sqrt(num_wires)))
        )
        feat_dim = self.features(dummy).shape[1]
        self.head = nn.Linear(feat_dim, num_wires)
        # Quantum encoder that maps the classical feature vector into the
        # computational basis of the quantum device.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.out_head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        classical = self.features(state_batch)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, classical)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.out_head(features).squeeze(-1)

__all__ = ["HybridQuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
