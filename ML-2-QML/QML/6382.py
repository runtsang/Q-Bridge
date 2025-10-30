"""Quantum‑enhanced Quanvolution network using PennyLane variational circuits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np

# Device that supports batched qubit simulation (CPU or GPU)
dev = qml.device("default.qubit", wires=4, shots=None)


def _quantum_circuit(params: np.ndarray, x_patch: np.ndarray) -> np.ndarray:
    """
    Parameterised quantum circuit that encodes a 2×2 image patch into a 4‑qubit state.
    Args:
        params: 2‑D array of shape (n_params, 4) containing rotation angles.
        x_patch: 1‑D array of shape (4,) with pixel intensities normalised to [0, 1].
    Returns:
        1‑D array of shape (4,) with expectation values of Pauli‑Z on each qubit.
    """
    # Encoding layer: rotate each qubit by the pixel intensity
    for i in range(4):
        qml.RY(x_patch[i], wires=i)
    # Variational layer
    for i in range(4):
        qml.RZ(params[0, i], wires=i)
        qml.RX(params[1, i], wires=i)
    # Entangling layer
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev, interface="torch")
def qnode(params: torch.Tensor, x_patches: torch.Tensor) -> torch.Tensor:
    """
    Batched quantum node that processes all patches in a single forward pass.
    Args:
        params: Tensor of shape (n_params, 4)
        x_patches: Tensor of shape (batch, 4)
    Returns:
        Tensor of shape (batch, 4) with measurement outcomes.
    """
    # Convert to numpy for the circuit function, then back to torch
    outputs = []
    for i in range(x_patches.shape[0]):
        out = _quantum_circuit(params.detach().cpu().numpy(), x_patches[i].detach().cpu().numpy())
        outputs.append(out)
    return torch.tensor(outputs, device=x_patches.device, dtype=torch.float32)


class QuanvolutionFilter(nn.Module):
    """
    Quantum filter that applies a shared variational circuit to every 2×2 patch of the input image.
    The variational parameters are trainable and shared across all patches.
    """

    def __init__(self, n_params: int = 2, dropout_prob: float = 0.1) -> None:
        super().__init__()
        # Parameters for the variational layer (RZ and RX per qubit)
        self.params = nn.Parameter(torch.randn(n_params, 4))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            Tensor of shape (batch, 4 * 14 * 14) containing quantum measurements.
        """
        batch_size = x.shape[0]
        # Create 2×2 patches and flatten them
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(batch_size, 14 * 14, 2, 2)
        patches = patches.permute(0, 1, 2, 3).reshape(batch_size, 14 * 14, 4)  # (batch, 196, 4)
        flat_patches = patches.reshape(-1, 4)  # (batch*196, 4)
        measurements = qnode(self.params, flat_patches)
        measurements = measurements.view(batch_size, 14 * 14, 4)
        measurements = self.dropout(measurements)
        return measurements.view(batch_size, -1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that concatenates the quantum filter output with
    a classical linear head. The linear layer can be regularised via
    weight decay during training.
    """

    def __init__(
        self,
        num_classes: int = 10,
        n_params: int = 2,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_params=n_params, dropout_prob=dropout_prob)
        self.linear = nn.Linear(4 * 14 * 14, num_classes, bias=True)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
