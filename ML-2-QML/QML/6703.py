"""Quanvolution classifier implemented with PennyLane variational circuits.

Each 2×2 pixel patch is encoded into a 4‑qubit circuit via Ry rotations,
passed through a two‑layer variational ansatz, and measured in the Z basis.
The resulting 4‑dimensional feature vector per patch is concatenated and fed
into a classical linear head. The model is fully differentiable using
PyTorch autograd and can be trained end‑to‑end with standard optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumCircuit(nn.Module):
    """Variational circuit that maps a 2×2 patch to 4 expectation values."""
    def __init__(self, n_qubits=4, n_layers=2, dev=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=0)
        # Parameters for the variational ansatz: shape (n_layers, n_qubits)
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta):
            # Encode inputs (pixel intensities) via Ry
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(theta[layer, qubit], wires=qubit)
                # Entangling CNOT chain
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            # Measure expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: (batch, 4)
        return self.circuit(patch, self.theta)


class QuanvolutionFilterQML(nn.Module):
    """Quantum filter that processes all patches of an image."""
    def __init__(self, use_quantum_kernel: bool = False):
        super().__init__()
        self.use_quantum_kernel = use_quantum_kernel
        self.qc = QuantumCircuit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        batch, _, height, width = x.shape
        # Reshape to patches: (batch, 14, 14, 4)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.squeeze(1)  # (batch, 14, 14, 2, 2)
        patches = patches.contiguous()
        patches = patches.view(-1, 4)  # (batch*14*14, 4)
        # Apply quantum circuit to each patch
        features = self.qc(patches)  # (batch*14*14, 4)
        return features


class QuanvolutionClassifier(nn.Module):
    """Hybrid quantum‑classical classifier."""
    def __init__(self, use_quantum_kernel: bool = False):
        super().__init__()
        self.qfilter = QuanvolutionFilterQML(use_quantum_kernel)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # (batch*14*14, 4)
        features = features.view(x.size(0), -1)  # (batch, 4*14*14)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilterQML", "QuanvolutionClassifier"]
