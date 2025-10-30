"""Quantum implementation of the hybrid network.

The network uses a quantum quanvolution filter to extract features from
2×2 patches and a variational quantum circuit that produces a single
expectation value.  The expectation is mapped to logits or a scalar
depending on the mode.  The same class can be used for binary
classification or regression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
import torchquantum as tq

class QuantumCircuit:
    """Wrapper around a parametrised single‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Two‑qubit quantum kernel applied to 2×2 image patches."""
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

class QuantumHybridNet(tq.QuantumModule):
    """
    Quantum‑classical hybrid network that first extracts features with a
    quantum quanvolution filter and then maps them through a variational
    quantum circuit to logits or a scalar.
    """
    def __init__(self, mode: str = "classification", shots: int = 200) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.mode = mode
        self.angle_mapper = nn.Linear(4 * 14 * 14, 1)
        backend = AerSimulator()
        self.quantum_circuit = QuantumCircuit(1, backend, shots=shots)
        if mode == "classification":
            self.out_layer = nn.Linear(1, 2)
        else:
            self.out_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        angles = self.angle_mapper(features).squeeze(-1).cpu().numpy()
        expectation = self.quantum_circuit.run(angles)
        expectation = torch.tensor(expectation, device=x.device, dtype=torch.float32)
        logits = self.out_layer(expectation.unsqueeze(-1))
        if self.mode == "classification":
            return torch.nn.functional.log_softmax(logits, dim=-1)
        return logits.squeeze(-1)

class QuantumRegressionDataset(Dataset):
    """Dataset that generates quantum states for regression."""
    def __init__(self, samples: int = 1000, num_wires: int = 4) -> None:
        self.states, self.labels = self._generate(samples, num_wires)

    @staticmethod
    def _generate(samples: int, num_wires: int) -> tuple[np.ndarray, np.ndarray]:
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

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumBinaryClassificationDataset(Dataset):
    """Dummy quantum binary classification dataset."""
    def __init__(self, samples: int = 1000, num_wires: int = 4) -> None:
        self.samples = samples
        self.num_wires = num_wires
        self.data = np.random.randn(samples, 2 ** num_wires).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=(samples,)).astype(np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self.samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.data[idx]),
            "label": torch.tensor(self.labels[idx]),
        }

__all__ = [
    "QuantumHybridNet",
    "QuantumRegressionDataset",
    "QuantumBinaryClassificationDataset",
]
