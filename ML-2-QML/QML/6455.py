import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    noise_std: float = 0.05,
    mix_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    The labels are constructed from the angles to provide a non‑trivial regression
    target that couples both amplitude and phase.
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
    labels += np.random.normal(scale=noise_std, size=labels.shape)

    # Mix classical and quantum contributions
    classical = np.random.uniform(-1.0, 1.0, size=samples)
    labels = mix_ratio * labels + (1 - mix_ratio) * classical
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary containing a complex quantum state
    and the regression target.
    """
    def __init__(self, samples: int, num_wires: int, mix_ratio: float = 0.5):
        self.states, self.labels = generate_superposition_data(num_wires, samples, mix_ratio=mix_ratio)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    A hybrid quantum‑classical regression model that uses a trainable
    variational circuit followed by a linear readout.  The encoder can
    be chosen at construction time, and the measurement layer is
    explicitly trainable through a learnable weight vector.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=40, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, encoding: str = "Ry"):
        super().__init__()
        self.n_wires = num_wires

        # Encoder selection
        if encoding == "Ry":
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
        elif encoding == "RxRy":
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRxRy"]
            )
        else:
            # Default to Ry if an unknown encoding is requested
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )

        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
