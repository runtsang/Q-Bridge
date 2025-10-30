"""QuantumRegression module – classical implementation.

This module re‑implements the seed regression example but adds a
hybrid quantum‑kernel pathway.  The public class `QuantumRegression`
accepts a ``mode`` argument that selects one of three training
behaviours:

* ``'ml'`` – pure classical MLP on real features.
* ``'kernel'`` – a quantum kernel is evaluated on the fly and the
  resulting similarity vector is fed into a linear head.
* ``'quantum'`` – the same variational circuit as in the QML seed is
  built with TorchQuantum and used for direct prediction.

The implementation keeps the original data‑generation routine and
dataset interface so that it can be dropped into the existing
scripts.  All components are fully PyTorch‑backed and therefore
compatible with standard optimisers and data loaders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq
from typing import Optional

# --------------------------------------------------------------------------- #
# 1. Data generation – identical to the ML seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return random features and a non‑linear target.

    The target is a smooth function of the sum of the features; it is
    deliberately simple so that the kernel can capture it.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# 2. Dataset – accepts a flag to return quantum states or classical
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Dataset that emits either real vectors or simulated quantum states."""
    def __init__(self, samples: int, num_features: int = 4, *, use_quantum: bool = False):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.use_quantum = use_quantum

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        if self.use_quantum:
            # Return complex representation for a quantum device
            return {
                "states": torch.tensor(self.features[index], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[index], dtype=torch.float32),
            }
        else:
            return {
                "states": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32),
            }

# --------------------------------------------------------------------------- #
# 3. Classical backbone – a small MLP
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Three‑layer feed‑forward network."""
    def __init__(self, input_dim: int, hidden: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# --------------------------------------------------------------------------- #
# 4. Quantum‑kernel module – uses TorchQuantum to evaluate a fixed ansatz
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Fixed quantum kernel that maps two real vectors into a similarity score."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires

    def _encode(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Encode a single real vector into the device."""
        for i, wire in enumerate(range(self.n_wires)):
            tq.RY(params=x[i], wires=wire)(q_device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap between the two encoded states."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires)
        qdev.reset_states(1)
        self._encode(qdev, x)
        state_x = qdev.states.clone()
        qdev.reset_states(1)
        self._encode(qdev, y)
        state_y = qdev.states.clone()
        overlap = torch.abs(torch.sum(state_x.conj() * state_y))
        return overlap

# --------------------------------------------------------------------------- #
# 5. Hybrid model – chooses between MLP, quantum kernel + head, or pure quantum
# --------------------------------------------------------------------------- #
class HybridModel(nn.Module):
    """A model that can act as a pure MLP, a quantum‑kernel head,
    or a fully quantum variational circuit.
    """
    def __init__(
        self,
        mode: str,
        num_features: int = 4,
        n_wires: int = 4,
        support_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.num_features = num_features
        self.n_wires = n_wires

        if self.mode == "ml":
            self.head = MLP(num_features)
        elif self.mode == "kernel":
            assert support_vectors is not None, "support_vectors required for kernel mode"
            self.kernel = QuantumKernel(n_wires)
            self.support_vectors = support_vectors
            self.head = nn.Linear(len(support_vectors), 1)
        elif self.mode == "quantum":
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
            self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.head = nn.Linear(n_wires, 1)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.mode == "ml":
            return self.head(batch)
        elif self.mode == "kernel":
            # Compute kernel vector between batch and support set
            batch = batch.to(torch.float32)
            support = self.support_vectors.to(torch.float32)
            kvec = torch.stack(
                [self.kernel(b, s) for s in support]
            ).transpose(0, 1)  # shape: (batch, support)
            return self.head(kvec)
        elif self.mode == "quantum":
            bsz = batch.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=batch.device)
            self.encoder(qdev, batch)
            self.q_layer(qdev)
            features = self.measure(qdev)
            return self.head(features).squeeze(-1)
        else:
            raise RuntimeError("Unreachable")

# --------------------------------------------------------------------------- #
# 6. Public wrapper that matches the original API
# --------------------------------------------------------------------------- #
class QuantumRegression(nn.Module):
    """
    Main entry point used by the rest of the repo.

    Parameters
    ----------
    mode : {'ml', 'kernel', 'quantum'}
        Select the training paradigm.
    num_features : int
        Dimensionality of the input data.
    n_wires : int
        Number of qubits used when ``mode`` is ``'quantum'`` or
        ``'kernel'``.
    support_vectors : torch.Tensor, optional
        Tensor of shape ``(k, num_features)`` used in kernel mode.
    """
    def __init__(
        self,
        mode: str = "ml",
        num_features: int = 4,
        n_wires: int = 4,
        support_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = HybridModel(
            mode=mode,
            num_features=num_features,
            n_wires=n_wires,
            support_vectors=support_vectors,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
