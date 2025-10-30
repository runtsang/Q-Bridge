"""Quantum regression model with a variational circuit and a linear head.

The model builds upon the classical seed but replaces the feed‑forward network
with a quantum variational circuit.  It supports arbitrary depth and a choice of
encoding (Ry, Rx, etc.).  The output is a scalar regression target obtained
by measuring all qubits in the Z basis and feeding the expectation values
into a classical linear layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_quantum_data(num_wires: int, samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.

    The target is a smooth function of the angles, mirroring the classical seed.
    """
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)

    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class QuantumHybridDataset(Dataset):
    """Dataset that returns quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_quantum_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegressionModel(tq.QuantumModule):
    """Variational quantum circuit for regression.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    depth : int, default 2
        Number of variational layers.
    encoder_type : str, default "Ry"
        Type of encoding gate (Ry, Rx, RZ, etc.).
    random_layer_ops : int, default 30
        Number of random two‑qubit gates in the random layer.
    """

    def __init__(
        self,
        num_wires: int,
        depth: int = 2,
        encoder_type: str = "Ry",
        random_layer_ops: int = 30,
    ) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth

        # Encoder that maps classical amplitudes to a quantum state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}x{encoder_type}"]
        )

        # Variational block
        class VariationalLayer(tq.QuantumModule):
            def __init__(self, n_wires: int, depth: int, ops_per_layer: int):
                super().__init__()
                self.n_wires = n_wires
                self.depth = depth
                self.random_layer = tq.RandomLayer(n_ops=ops_per_layer, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice) -> None:
                self.random_layer(qdev)
                for _ in range(self.depth):
                    for wire in range(self.n_wires):
                        self.rx(qdev, wires=wire)
                        self.ry(qdev, wires=wire)

        self.var_layer = VariationalLayer(num_wires, depth, random_layer_ops)

        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "QuantumHybridDataset", "generate_quantum_data"]
