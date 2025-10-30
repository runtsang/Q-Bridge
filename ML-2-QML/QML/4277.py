"""Hybrid quantum‑classical classifier combining a quanvolution filter with a Qiskit EstimatorQNN.

The module implements a quantum circuit that processes each 2×2 image patch using
parameterised Ry and Rx gates followed by a small entangling layer.  The circuit
is wrapped in a Qiskit EstimatorQNN, which evaluates the expectation value of
four Pauli‑Z observables (one per qubit).  The resulting features are then fed
into a classical linear classifier.  The design is inspired by the original
quanvolution filter, the quantum regression dataset, and the EstimatorQNN
example.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.estimators import StatevectorEstimator

__all__ = [
    "QuanvolutionHybridQuantum",
    "QuanvolutionFilterQuantum",
    "generate_superposition_data",
    "RegressionDataset",
]

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition state dataset for the quantum regression example."""
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

class RegressionDataset(data.Dataset):
    """Dataset wrapping the synthetic quantum regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class PatchQuantumCircuit:
    """Quantum circuit that encodes a 2×2 image patch into a 4‑qubit state.

    The circuit uses Ry gates to encode the pixel intensities and Rx gates as
    trainable weights.  A small entangling layer is added to increase expressivity.
    """

    def __init__(self):
        # Input parameters (pixel values)
        self.px = [Parameter(f"px_{i}") for i in range(4)]
        # Weight parameters (trainable)
        self.wt = [Parameter(f"wt_{i}") for i in range(4)]

        # Build the circuit
        self.circuit = QuantumCircuit(4)
        for i in range(4):
            self.circuit.h(i)
            self.circuit.ry(self.px[i], i)
            self.circuit.rx(self.wt[i], i)

        # Simple entangling layer: a chain of CNOTs
        for i in range(3):
            self.circuit.cx(i, i + 1)

        # Measurement observables: Pauli‑Z on each qubit
        self.observables = [
            Pauli.from_label("Z" + "I" * 3),
            Pauli.from_label("I" + "Z" + "I" * 2),
            Pauli.from_label("I" * 2 + "Z" + "I"),
            Pauli.from_label("I" * 3 + "Z"),
        ]

class QuanvolutionFilterQuantum(nn.Module):
    """Quantum quanvolution filter that processes each 2×2 patch with a PatchQuantumCircuit.

    The filter returns a feature vector of expectation values for all patches in the image.
    """

    def __init__(self):
        super().__init__()
        self.patch_circuit = PatchQuantumCircuit()
        self.simulator = AerSimulator(method="statevector")
        # Use a StatevectorEstimator to evaluate expectation values
        self.estimator = StatevectorEstimator(self.simulator)
        # Build an EstimatorQNN wrapper for efficient batched evaluation
        self.qnn = EstimatorQNN(
            circuit=self.patch_circuit.circuit,
            observables=self.patch_circuit.observables,
            input_params=self.patch_circuit.px,
            weight_params=self.patch_circuit.wt,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter to a batch of images.

        Args:
            x: Tensor of shape (B, 1, 28, 28) with pixel intensities in [-1, 1].

        Returns:
            Tensor of shape (B, 14*14*4) containing the expectation values for each patch.
        """
        bsz = x.shape[0]
        # Reshape to patches: (B, 14, 14, 4)
        patches = x.view(bsz, 1, 28, 28).unfold(2, 2, 2).unfold(3, 2, 2)
        patches = patches.permute(0, 2, 3, 4, 5).contiguous()  # (B, 14, 14, 2, 2)
        patches = patches.reshape(-1, 4)  # (B*14*14, 4)

        # Convert to numpy for binding
        patch_np = patches.cpu().numpy()

        # Estimate expectation values for all patches
        exp_vals = self.qnn(patch_np)  # shape (B*14*14, 4)
        exp_vals = torch.tensor(exp_vals, dtype=torch.float32, device=x.device)
        exp_vals = exp_vals.view(bsz, 14, 14, 4)  # (B, 14, 14, 4)
        return exp_vals.view(bsz, -1)  # flatten to (B, 14*14*4)

class QuanvolutionHybridQuantum(nn.Module):
    """Hybrid quantum‑classical classifier.

    The model applies a quantum quanvolution filter to extract patch‑wise features,
    then feeds the flattened feature vector into a classical linear classifier.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.classifier = nn.Linear(14 * 14 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)
