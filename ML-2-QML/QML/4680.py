"""Quantum kernel module with TorchQuantum and SamplerQNN integration.

This module implements a variational circuit that encodes two‑dimensional
classical data and returns a kernel value.  It is designed to be called from
the classical layer via a callable interface.  The circuit uses a mixed
encoding: an Ry rotation for each input followed by a small entangling block.
The SamplerQNN network is used as a pre‑processing step to produce a richer
feature representation before the quantum kernel evaluation.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# SamplerQNN quantum circuit (seed 2, quantum part)
# --------------------------------------------------------------------------- #
def quantum_sampler_qnn() -> tq.QuantumModule:
    """Return a QuantumModule that encodes a two‑qubit state.

    The circuit implements the Qiskit example from reference[2] but in TorchQuantum
    syntax.  It is used as a feature extractor before the main kernel circuit.
    """
    class _SamplerQuantum(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.cx = tq.CX
            self.ryw = [tq.RY(has_params=True, trainable=True) for _ in range(2)]

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.ry0(qdev, wires=0)
            self.ry1(qdev, wires=1)
            self.cx(qdev, wires=[0, 1])
            for i, ry in enumerate(self.ryw):
                ry(qdev, wires=i)

    return _SamplerQuantum()

# --------------------------------------------------------------------------- #
# Hybrid quantum kernel (seed 1, quantum part)
# --------------------------------------------------------------------------- #
class HybridQuantumKernel(tq.QuantumModule):
    """Variational quantum kernel that accepts two classical inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple Ry encoding for 4 inputs
        self.ry_enc = [tq.RY(has_params=True, trainable=True) for _ in range(self.n_wires)]
        # Entanglement pattern
        self.cx = tq.CX
        # Pre‑processing sampler
        self.sampler = quantum_sampler_qnn()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return |<x|y>|^2 as a kernel value."""
        # Ensure proper shape
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        # Encode x
        for i, ry in enumerate(self.ry_enc):
            ry(self.device, wires=[i], params=x[0, i])
        # Run sampler (acts as a feature map)
        self.sampler(self.device)
        # Unwind with y (negative parameters)
        for i, ry in reversed(list(enumerate(self.ry_enc))):
            ry(self.device, wires=[i], params=-y[0, i])
        # Compute overlap
        return torch.abs(self.device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# Quantum kernel matrix
# --------------------------------------------------------------------------- #
def quantum_kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
) -> np.ndarray:
    """Compute the Gram matrix using the HybridQuantumKernel."""
    kernel = HybridQuantumKernel()
    return np.array(
        [[kernel(x, y).item() for y in b] for x in a]
    )

# --------------------------------------------------------------------------- #
# Quantum regression dataset (seed 3, quantum part)
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

# --------------------------------------------------------------------------- #
# Quantum regression model (seed 3, quantum part)
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(
                n_ops=30, wires=list(range(num_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=[wire])
                self.ry(qdev, wires=[wire])

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
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
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "QuantumSamplerQNN",
    "HybridQuantumKernel",
    "quantum_kernel_matrix",
    "generate_superposition_data",
    "QuantumRegressionModel",
]
