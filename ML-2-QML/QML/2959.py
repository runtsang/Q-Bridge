"""Hybrid regression module with a quantum kernel front‑end.

The implementation mirrors the classical module but replaces the feature
extraction stage with a parameterised quantum circuit.  The circuit
encodes the input data into a superposition state, applies a random
variational layer and single‑qubit rotations, and measures all qubits
in the Pauli‑Z basis.  The resulting expectation values are then fed
into a linear read‑out.  An optional quantum kernel module can be
used to produce kernelised features for the regression head,
demonstrating how a classical kernel can be replaced by a quantum one
while keeping the rest of the pipeline unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Sequence

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition samples for a quantum device.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the circuit.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states, labels : tuple[np.ndarray, np.ndarray]
        ``states`` is a ``(samples, 2**num_wires)`` array of complex amplitudes
        representing a superposition of |0…0⟩ and |1…1⟩ with random angles.
        ``labels`` are a noisy target derived from the angles.
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
    return states, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RegressionDataset(torch.utils.data.Dataset):
    """Torch dataset wrapping the quantum superposition data.

    The dataset yields a dictionary with keys ``states`` (complex
    amplitude arrays of shape ``(2**num_wires,)``) and ``target`` (float).
    """

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum kernel
# --------------------------------------------------------------------------- #

class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz that evaluates a simple quantum kernel.

    The circuit consists of a set of single‑qubit Ry gates followed by
    their inverse applied to a second copy of the input data.  The
    overlap of the resulting state with the initial state is returned
    as a real number between 0 and 1.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a pair of inputs."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(1)
        self.ansatz(self.q_device, x)
        self.ansatz(self.q_device, -y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute a Gram matrix using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression head using a variational encoder.

    The model consists of:
      * a data encoder that maps the classical input to a quantum state,
      * a random variational layer followed by trainable RX/RY rotations,
      * measurement of all qubits in the Pauli‑Z basis,
      * a linear read‑out that produces the regression target.
    """

    class QLayer(tq.QuantumModule):
        """Variational sub‑module with a random layer and single‑qubit rotations."""

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, use_kernel: bool = False) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.use_kernel = use_kernel

        # Encoder maps classical vector to quantum state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.num_wires}xRy"]
        )
        self.q_layer = self.QLayer(self.num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.num_wires, 1)

        if use_kernel:
            # Keep a support set of states for the quantum kernel
            self.support_states = None

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass that optionally uses a quantum kernel."""

        batch_size = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=batch_size, device=state_batch.device)

        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)

        if self.use_kernel:
            # Compute the kernel between the batch and the support set
            if self.support_states is None:
                # lazily initialise support set to the batch itself
                self.support_states = state_batch.clone().detach()
            kernel_matrix = torch.matmul(features, features.T)
            # Linear read‑out on kernel features
            return self.head(kernel_matrix).squeeze(-1)
        else:
            return self.head(features).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumKernel",
    "quantum_kernel_matrix",
    "HybridRegressionModel",
]
