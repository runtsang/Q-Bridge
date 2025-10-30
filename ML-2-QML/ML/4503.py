"""Hybrid fraud detection model combining classical convolutional layers, a photonic‑inspired feature map, and a quantum fully connected layer.

The model integrates concepts from:
- FraudDetection (photonic layers)
- QCNN (convolutional structure)
- QuantumNAT (quantum fully connected layer)
- FCL (fully connected layer)
"""

from __future__ import annotations

import torch
from torch import nn
import qiskit
from qiskit import QuantumCircuit
from typing import Tuple

# Import the quantum circuit builder from the quantum module
from fraud_detection_quantum import build_quantum_circuit

class QuantumWrapper(nn.Module):
    """Executes a Qiskit circuit and returns expectation values of PauliZ."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.circuit = build_quantum_circuit(n_qubits)
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params shape: (batch, 4*n_qubits)
        batch, total = params.shape
        if total!= 4 * self.n_qubits:
            raise ValueError(f"Expected {4*self.n_qubits} parameters per sample, got {total}")
        # Split parameters
        fm = params[:, :self.n_qubits]
        conv = params[:, self.n_qubits:2*self.n_qubits]
        pool = params[:, 2*self.n_qubits:3*self.n_qubits]
        qfc = params[:, 3*self.n_qubits:4*self.n_qubits]
        # Build parameter binds
        param_binds = []
        for i in range(batch):
            bind = {}
            for j in range(self.n_qubits):
                bind[f"fm_{j}"] = float(fm[i, j])
                bind[f"conv_{j}"] = float(conv[i, j])
                bind[f"pool_{j}"] = float(pool[i, j])
                bind[f"qfc_{j}"] = float(qfc[i, j])
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, backend=self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        expectations = []
        for i in range(batch):
            counts = result.get_counts(index=i)
            exp_per_qubit = []
            for q in range(self.n_qubits):
                exp = 0.0
                for bitstring, cnt in counts.items():
                    bit = int(bitstring[::-1][q])
                    exp += ((-1) ** bit) * cnt
                exp /= self.shots
                exp_per_qubit.append(exp)
            expectations.append(exp_per_qubit)
        return torch.tensor(expectations, dtype=torch.float32, device=params.device)

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model."""
    def __init__(self, in_channels: int = 1, n_qubits: int = 4):
        super().__init__()
        # Classical feature extractor (QCNN‑like)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU()
        )
        # Quantum layer
        self.quantum_layer = QuantumWrapper(n_qubits=n_qubits)
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        features = self.feature_extractor(x)
        # Prepare quantum parameters: use first 4*n_qubits features
        q_params = features[:, :4 * self.quantum_layer.n_qubits]
        q_out = self.quantum_layer(q_params)
        # Concatenate classical and quantum outputs
        combined = torch.cat([features, q_out], dim=1)
        return self.classifier(combined)

__all__ = ["FraudDetectionHybrid"]
