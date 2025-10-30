"""Hybrid regression model – quantum implementation.

The quantum version uses a general encoder, a RandomLayer, Rx/Ry gates,
and measures Pauli‑Z on all wires.  The resulting expectation values
are fed into a linear head, mirroring the classical model’s structure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Iterable, List

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and regression targets.

    States are of the form cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
    The target is sin(2 theta) * cos(phi), matching the classical target
    function up to a change of variables.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning quantum state vectors and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum hybrid head – encoder + random layer + Rx/Ry + measurement
# --------------------------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression head with a linear post‑processing layer.

    The circuit is built from a general encoder that maps the input
    amplitudes into a computational basis, a RandomLayer that
    injects trainable parameters, and a sequence of Rx/Ry gates
    applied to each wire.  After measuring Pauli‑Z on all wires we
    feed the expectation values into a linear layer.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int, hidden_dim: int = 32):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that converts the amplitude vector into a basis state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hybrid circuit.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, 2**n_wires) of complex amplitudes.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Estimator – matches the FastBaseEstimator pattern
# --------------------------------------------------------------------------- #
class FastEstimator:
    """Evaluate the quantum hybrid model on a batch of state vectors.

    The implementation is identical in spirit to the classical
    FastEstimator but operates on complex tensors and uses the
    quantum device internally.
    """
    def __init__(self, model: tq.QuantumModule):
        self.model = model

    def evaluate(
        self,
        inputs: Iterable[Iterable[complex]],
        *,
        device: torch.device | str = "cpu",
    ) -> List[float]:
        """Return real predictions for each input state."""
        self.model.eval()
        with torch.no_grad():
            batch = torch.as_tensor(inputs, dtype=torch.cfloat, device=device)
            preds = self.model(batch).cpu().numpy()
        return preds.tolist()

__all__ = ["RegressionDataset", "HybridRegressionModel", "FastEstimator"]
