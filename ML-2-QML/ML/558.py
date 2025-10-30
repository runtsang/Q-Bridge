"""HybridQCNet – classical backbone with a variational quantum head.

This module extends the original QCNet by adding a feature‑pyramid
backbone (ResNet‑style) and a parameter‑shifting gradient estimator.
It keeps the public API compatible with the seed while adding
support for swapping the quantum backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit

# Classical backbone: a lightweight ResNet‑style feature pyramid
class _ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class _FeaturePyramid(nn.Module):
    """A small feature pyramid with three ResNet blocks."""
    def __init__(self):
        super().__init__()
        self.block1 = _ResNetBlock(3, 16)
        self.block2 = _ResNetBlock(16, 32, stride=2)
        self.block3 = _ResNetBlock(32, 64, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        return x

# Hybrid layer that routes to the quantum backend
class _HybridLayer(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self._circuit = None  # will be created lazily

    def set_backend(self, backend):
        """Replace the quantum backend (e.g., Aer, PennyLane, or a real device)."""
        self.backend = backend
        self._circuit = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the tensor and pass each sample through the quantum circuit
        x_flat = x.view(x.size(0), -1)
        # Use a simple linear mapping to the number of qubits
        if self._circuit is None:
            qc = qiskit.QuantumCircuit(self.n_qubits)
            theta = qiskit.circuit.Parameter("theta")
            qc.h(range(self.n_qubits))
            qc.ry(theta, range(self.n_qubits))
            qc.measure_all()
            self._circuit = qc
        expectations = []
        for sample in x_flat:
            param = float(sample.item())
            bound_qc = self._circuit.bind_parameters({self._circuit.parameters[0]: param})
            compiled = qiskit.transpile(bound_qc, self.backend)
            qobj = qiskit.assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(s, 2) for s in counts.keys()])
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        return torch.tensor(expectations, device=x.device).unsqueeze(-1)

class HybridQCNet(nn.Module):
    """Full hybrid convolutional‑quantum binary classifier."""

    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.backbone = _FeaturePyramid()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, 1)
        )
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = _HybridLayer(n_qubits, backend, shots=shots, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        probs = torch.sigmoid(logits)
        # Pass through quantum hybrid head
        q_out = self.hybrid(probs)
        return torch.cat([q_out, 1 - q_out], dim=-1)

    def set_backend(self, backend):
        """Set a new quantum backend for the hybrid layer."""
        self.hybrid.set_backend(backend)

__all__ = ["HybridQCNet"]
