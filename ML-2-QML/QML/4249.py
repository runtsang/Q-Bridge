from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.primitives import Sampler
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

import torchquantum as tq


class QuantumSamplerCircuit:
    """Parameterized two‑qubit circuit used by the quantum head."""
    def __init__(self, backend, shots: int = 200):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Input encoding
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        # Entanglement
        self.circuit.cx(0, 1)
        # Variational block
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        self.circuit.measure_all()
        self.sampler = Sampler(backend=self.backend, shots=self.shots)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of input parameters."""
        batch = []
        for inp in inputs:
            bind = {self.input_params[0]: inp[0], self.input_params[1]: inp[1]}
            job = self.sampler.run(self.circuit, bind)
            result = job.result()
            counts = result.get_counts()
            # Map counts to probabilities of first‑qubit being 0
            prob0 = sum(v for k, v in counts.items() if k[0] == '0') / self.shots
            prob1 = 1.0 - prob0
            batch.append([prob0, prob1])
        return np.array(batch, dtype=np.float32)


class QuantumSamplerHead(nn.Module):
    """Hybrid head that maps classical features to input parameters of the
    quantum sampler and returns a two‑class probability vector."""
    def __init__(self, input_dim: int, backend, shots: int = 200) -> None:
        super().__init__()
        self.input_mapping = nn.Linear(input_dim, 2)
        self.quantum_circuit = QuantumSamplerCircuit(backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map features to 2 input parameters
        params = self.input_mapping(x).detach().cpu().numpy()
        # Run quantum sampler
        probs = self.quantum_circuit.run(params)
        return torch.tensor(probs, device=x.device, dtype=x.dtype)


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 patches of a grayscale image."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridQuantumBinaryClassifier(nn.Module):
    """Quantum‑classical hybrid binary classifier that combines a
    convolutional backbone, a quantum quanvolution filter, and a quantum
    sampler head."""
    def __init__(self, backend: qiskit.providers.BaseBackend | None = None) -> None:
        super().__init__()
        if backend is None:
            backend = AerSimulator()
        # Quantum quanvolution filter
        self.qfilter = QuanvolutionFilter()
        # Back‑bone conv layers processing the quanvolution output
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten()
        )
        # Quantum sampler head
        self.quantum_head = QuantumSamplerHead(4 * 14 * 14, backend, shots=200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to grayscale and apply quanvolution
        gray = x.mean(dim=1, keepdim=True)
        qfeat = self.qfilter(gray)
        # Reshape to 4‑channel feature map of size 14×14
        qfeat_map = qfeat.view(x.size(0), 4, 14, 14)
        # Classical convolutional processing
        feat = self.backbone(qfeat_map)
        # Quantum sampler head returns probabilities
        probs = self.quantum_head(feat)
        return probs

__all__ = ["HybridQuantumBinaryClassifier", "QuantumSamplerCircuit",
           "QuantumSamplerHead", "QuanvolutionFilter"]
