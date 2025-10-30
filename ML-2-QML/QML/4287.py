import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum implementation of the hybrid model.
    The circuit encodes a 2×2 image patch into a 4‑qubit register,
    applies a random variational layer, and measures all qubits.
    A linear head converts the measurement vector to class logits
    or a regression output.
    """
    def __init__(self, num_classes: int = 10, regression: bool = False) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder maps the 4‑pixel patch into the 4‑qubit state
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        out_dim = 1 if regression else num_classes
        self.head = nn.Linear(self.n_wires, out_dim)

    class QLayer(tq.QuantumModule):
        """Random variational layer followed by parameterised rotations."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ``x`` is expected to be a batch of 1×28×28 images.
        The filter is applied patch‑wise, but for simplicity we
        treat the entire image as a single 4‑qubit block.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Flatten the image to a 4‑pixel vector per batch element
        patches = x.view(bsz, -1)[:, :4]
        self.encoder(qdev, patches)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
# Auxiliary utilities – quantum regression dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0…0⟩ + e^{iφ} sin(theta)|1…1⟩
    and a target label derived from theta and phi.
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Quantum regression dataset yielding a state vector and a scalar target."""
    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["QuanvolutionHybrid", "RegressionDataset", "generate_superposition_data"]
