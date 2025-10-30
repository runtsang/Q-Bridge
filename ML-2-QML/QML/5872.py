"""
QuanvolutionHybridQuantumClassifier – a quantum‑centric model that implements the same pipeline as the classical hybrid, but executes the filter and fully‑connected layer on a quantum simulator using PennyLane.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumQuanvolutionFilter(qml.QuantumNode):
    """
    Quantum filter that encodes 2×2 image patches into 4 qubits, applies a random circuit,
    and measures in the Z basis. The circuit is re‑used for every patch.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1000, dev_name: str = "default.qubit") -> None:
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)
        self.n_qubits = n_qubits
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(patch):
            # Encode patch values into Ry rotations
            for i, val in enumerate(patch):
                qml.RY(val, wires=i)
            # Random entangling layer
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.circuit = circuit

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        # patch shape: (batch, 4)
        return self.circuit(patch)


class QuantumFullyConnectedLayer(nn.Module):
    """
    Parameterized quantum circuit acting as a fully‑connected layer.
    Uses a single qubit with a Ry gate parameterized by the input and returns the expectation value.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1000, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1)
        return self.circuit(x).unsqueeze(-1)


class QuanvolutionHybridQuantumClassifier(nn.Module):
    """
    Full quantum‑hybrid classifier:
        1. Classical extraction of 2×2 patches.
        2. Quantum quanvolution filter applied to each patch.
        3. Quantum fully‑connected layer per patch.
        4. Classical linear head to produce logits.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 patch_size: int = 2, stride: int = 2,
                 shots: int = 1000, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_patches = (28 // stride) ** 2
        self.quantum_filter = QuantumQuanvolutionFilter(n_qubits=4, shots=shots, dev_name=dev_name)
        self.quantum_fc = QuantumFullyConnectedLayer(n_qubits=1, shots=shots, dev_name=dev_name)
        self.classifier = nn.Linear(self.n_patches, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        """
        batch_size = x.size(0)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)  # shape: (batch, 4, n_patches)
        patches = patches.permute(0, 2, 1)  # (batch, n_patches, 4)
        # Flatten to process all patches in a single batch
        patches_flat = patches.reshape(-1, 4)  # (batch * n_patches, 4)
        # Quantum filter
        q_features = self.quantum_filter(patches_flat)  # (batch * n_patches, 4)
        # Quantum fully‑connected layer on each patch feature
        q_fc_out = self.quantum_fc(q_features.mean(dim=1, keepdim=True))  # (batch * n_patches, 1)
        q_fc_out = q_fc_out.reshape(batch_size, self.n_patches)  # (batch, n_patches)
        logits = self.classifier(q_fc_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridQuantumClassifier", "QuantumQuanvolutionFilter", "QuantumFullyConnectedLayer"]
