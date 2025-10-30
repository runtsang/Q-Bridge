"""Quantum‑class hybrid model that mirrors :class:`QuanvolutionHybrid` (classical version).

It replaces the classical convolution and RBF kernel with a quanvolutional filter and a quantum kernel,
respectively, and provides synthetic regression utilities adapted to complex‑state data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
from typing import Sequence

# ----------------------------------------------------------------------
# Quantum quanvolution filter
# ----------------------------------------------------------------------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# ----------------------------------------------------------------------
# Quantum kernel utilities (from the reference)
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
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
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
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

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Hybrid quantum model
# ----------------------------------------------------------------------
class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum hybrid model mirroring :class:`QuanvolutionHybrid` (classical version).
    * Quantum convolution (quanvolution) extracts 2×2 patches.
    * Quantum kernel evaluates similarity to a fixed set of prototypes.
    * Linear head produces logits.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.kernel = Kernel()
        self.prototypes = None  # to be set via ``set_prototypes``.
        self.head = nn.Linear(0, num_classes)  # placeholder; re‑initialised after prototypes.

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """
        Register a fixed set of prototype vectors against which the quantum kernel is evaluated.
        Parameters
        ----------
        prototypes : torch.Tensor
            Shape (P, D) where P is the number of prototypes and D is the feature dimension.
        """
        self.prototypes = prototypes
        self.head = nn.Linear(prototypes.shape[0], self.head.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.prototypes is None:
            raise RuntimeError("Prototypes not set. Call ``set_prototypes`` before forward.")
        feats = self.qfilter(x)  # (B, D)
        B, D = feats.shape
        P = self.prototypes.shape[0]
        # Compute kernel matrix via nested loops (small prototype sets are expected).
        kernel_vals = torch.zeros(B, P, device=feats.device)
        for i in range(B):
            for j in range(P):
                kernel_vals[i, j] = self.kernel(feats[i].unsqueeze(0), self.prototypes[j].unsqueeze(0))
        logits = self.head(kernel_vals)
        return F.log_softmax(logits, dim=-1)

# ----------------------------------------------------------------------
# Synthetic regression utilities (quantum version)
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Dataset wrapping the quantum synthetic regression data."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = [
    "QuanvolutionHybrid",
    "QuantumQuanvolutionFilter",
    "Kernel",
    "quantum_kernel_matrix",
    "generate_superposition_data",
    "RegressionDataset",
]
