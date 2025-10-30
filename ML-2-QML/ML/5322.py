from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence

# ------------------------------------------------------------------
# Dataset – physics‑based superposition with phase noise
# ------------------------------------------------------------------
def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    amplitude_noise: float = 0.05,
    phase_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    # Create a batch of states that are linear combinations of |0...0> and |1...1>.
    # Noise terms are added to both the amplitude and the phase so the model
    # learns to be robust to realistic imperfections.
    thetas = 2 * np.pi * np.random.rand(samples)
    phis   = 2 * np.pi * np.random.rand(samples)

    zero_state = np.zeros(2 ** num_features, dtype=complex)
    zero_state[0] = 1.0
    one_state  = np.zeros(2 ** num_features, dtype=complex)
    one_state[-1] = 1.0
    states = (
        np.cos(thetas)[:, None] * zero_state
        + np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * one_state
    )

    labels = np.sin(2 * thetas) * np.cos(phis)
    labels += amplitude_noise * np.random.randn(samples)
    return states.astype(np.complex128), labels.astype(np.float32)

class RegressionDataset(Dataset):
    # A Dataset that returns a complex‑valued state and a real‑valued target.
    # Mirrors the QML interface but keeps all data in PyTorch tensors for
    # compatibility with the classical head.
    def __init__(self, samples: int, num_features: int):
        self.states, self.targets = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
           'states': torch.tensor(self.states[idx], dtype=torch.cfloat),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Classical core – deep MLP
# ------------------------------------------------------------------
class ClassicalMLP(nn.Module):
    # A flexible MLP that can be stacked on top of the quantum encoder.
    # The architecture is intentionally deep to allow the network to learn
    # the residual between the quantum and classical predictions.
    def __init__(self, in_features: int, hidden_dims: Sequence[int] = (64, 32, 16)):
        super().__init__()
        layers = []
        input_dim = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ------------------------------------------------------------------
# Hybrid quantum‑classical head
# ------------------------------------------------------------------
import torchquantum as tq

class HybridQubitHead(tq.QuantumModule):
    # Quantum core that uses a reusable QuantumDevice and a RandomLayer
    # to generate a feature vector.  The RandomLayer is trainable and
    # the final measurement head is a linear layer that maps the
    # **features** (‑> 1 output) which is to be combined with the
    # **classical MLP**.
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f'{num_wires}xRy']
        )
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# ------------------------------------------------------------------
# Full model – hybrid regression
# ------------------------------------------------------------------
class QuantumRegressionGen(nn.Module):
    # Combines the quantum encoder + hybrid head with a deep classical MLP
    # to produce a single scalar output.  The two branches are summed
    # and the result is trained with MSE loss.
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.q_head = HybridQubitHead(num_wires)
        self.classical = ClassicalMLP(num_features)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        q_out = self.q_head(state_batch)
        c_out = self.classical(state_batch)
        return q_out + c_out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (n_wires={self.q_head.q_layer.n_wires})'

__all__ = ['QuantumRegressionGen', 'RegressionDataset', 'generate_superposition_data']
