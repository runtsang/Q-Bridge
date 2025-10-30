"""
Quantum hybrid quanvolutional network with a 4‑qubit variational block.

The network processes 2×2 image patches with an Ry encoding, passes them
through a random layer followed by trainable RX/RY rotations, and measures
all qubits.  A linear head maps the measurement outcomes to either
log‑softmax logits or regression outputs.  The module also provides
utilities to generate superposition data for regression experiments.
"""

import torch
import torch.nn as nn
import torchquantum as tq
from typing import Iterable, Tuple, List
import numpy as np


def generate_superposition_data(
    num_wires: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    Labels are sin(2θ) * cos(φ), mirroring the regression example.
    """
    omega0 = np.zeros(2**num_wires, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(2**num_wires, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum quanvolutional backbone + linear head.

    Parameters
    ----------
    task : str, default='classify'
        'classify' or'regress'.
    n_classes : int, default=10
        Number of classes for classification. Ignored for regression.
    """

    class _QLayer(tq.QuantumModule):
        """Variational block: random layer + trainable RX/RY rotations."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, task: str = "classify", n_classes: int = 10) -> None:
        super().__init__()
        self.task = task
        self.n_wires = 4

        # Encode each pixel with an Ry gate
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.qlayer = self._QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        if self.task == "classify":
            self.head = nn.Linear(self.n_wires, n_classes)
        else:
            self.head = nn.Linear(self.n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

        # Reshape to 2×2 patches
        img = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [img[:, r, c], img[:, r, c + 1],
                     img[:, r + 1, c], img[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.qlayer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))

        features = torch.cat(patches, dim=1)
        logits = self.head(features)

        if self.task == "classify":
            return F.log_softmax(logits, dim=-1)
        else:
            return logits.squeeze(-1)

    def set_task(self, task: str, n_classes: int = 10) -> None:
        if task not in ("classify", "regress"):
            raise ValueError('task must be "classify" or "regress"')
        self.task = task
        if task == "classify":
            self.head = nn.Linear(self.n_wires, n_classes)
        else:
            self.head = nn.Linear(self.n_wires, 1)


__all__ = ["QuanvolutionHybrid", "generate_superposition_data"]
