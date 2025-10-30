"""Quantum‑aware quanvolution using PennyLane.

The quantum filter replaces the 2×2 classical convolution with a
parameterised 4‑qubit circuit.  Each 2×2 pixel patch is encoded
via rotation gates, entangled with CNOTs, and measured in the Pauli‑Z
basis.  The resulting four expectation values form the feature map
which is then passed to the same linear classifier head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np  # type: ignore
from typing import List


class QuantumPatchCircuit:
    """
    Parameterised 4‑qubit circuit used as a kernel for each 2×2 patch.
    The circuit consists of:
      - 3 rotation layers (Rx, Ry, Rz) per qubit
      - a linear entangling layer of CNOTs
    The parameters are learned jointly with the classical head.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        # Number of parameters: 3 * n_qubits
        self.n_params = 3 * n_qubits
        # PennyLane device (default to CPU simulator)
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=1024)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode the four pixel values as Ry rotations
            for i in range(self.n_qubits):
                qml.Ry(inputs[i], wires=i)
            # Parameterised rotation layer
            for i in range(self.n_qubits):
                qml.Rx(params[3 * i + 0], wires=i)
                qml.Ry(params[3 * i + 1], wires=i)
                qml.Rz(params[3 * i + 2], wires=i)
            # Entangling CNOT chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def __call__(self, patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: Tensor of shape (4,) containing the 2×2 pixel values.
            params: Tensor of shape (n_params,) containing trainable parameters.
        Returns:
            Tensor of shape (4,) with expectation values.
        """
        return self.circuit(patch, params)


class QuanvolutionFilterQuantum(nn.Module):
    """Quantum version of the quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.n_qubits = 4
        self.circuit = QuantumPatchCircuit(n_qubits=self.n_qubits)
        # Learnable parameters for the circuit
        self.params = nn.Parameter(torch.randn(self.circuit.n_params))
        # Linear layer to map the 4‑dimensional output to a scalar per patch
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 28, 28)
        Returns:
            Tensor of shape (B, 4*14*14) – flattened feature map.
        """
        B = x.size(0)
        # Reshape into patches: (B, 14, 14, 4)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(B, 1, 14, 14, 4)  # (B, 1, 14, 14, 4)
        # Move channel to batch for batch processing
        patches = patches.permute(0, 1, 2, 3, 4)  # (B, 1, 14, 14, 4)
        feats = []
        for i in range(14):
            row_feats = []
            for j in range(14):
                # Extract 4‑pixel patch
                patch = patches[:, :, i, j, :].squeeze(1)  # (B, 4)
                # Apply quantum circuit
                q_out = self.circuit(patch, self.params)  # (B, 4)
                # Linear projection to a single feature
                proj = self.linear(q_out)  # (B, 1)
                row_feats.append(proj)
            # Concatenate across columns
            row = torch.cat(row_feats, dim=1)  # (B, 14)
            feats.append(row)
        # Concatenate across rows
        feat_map = torch.cat(feats, dim=1)  # (B, 14*14)
        return feat_map.view(B, -1)  # flatten to (B, 4*14*14)


class QuanvolutionClassifierQuantum(nn.Module):
    """Hybrid classifier using the quantum quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilterQuantum", "QuanvolutionClassifierQuantum"]
