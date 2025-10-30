"""Quantum regression with a convolution‑style measurement filter.

The quantum model encodes classical 2‑D data into a quantum device,
applies a parameterized rotation layer, a random unitary, and then
measures a custom convolution filter implemented with a random
circuit.  The measurement results are mapped to a regression output
via a linear layer."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Same data generator as the classical counterpart but returns NumPy arrays
    for compatibility with Qiskit.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Wraps the data generator and reshapes it into 2‑D images."""
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)
        side = math.isqrt(num_features)
        if side * side!= num_features:
            pad = side * side - num_features
            self.states = np.pad(self.states, ((0, 0), (0, pad)))
        self.states = self.states.reshape(-1, side, side)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"states": torch.tensor(self.states[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QuantumConvFilter:
    """
    Implements a convolution‑style measurement using a random circuit.
    Each qubit corresponds to a pixel in the kernel window.
    """
    def __init__(self, kernel_size: int, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Parameterized RX gates
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = data.flatten()
        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0}
                       for i, val in enumerate(data_flat)]
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        # Count total number of ones over all shots
        total_ones = sum(sum(int(bit) for bit in key) * count
                         for key, count in result.items())
        return total_ones / (self.shots * self.n_qubits)

class QRegressor(nn.Module):
    """
    Variational quantum regression model that uses a convolution‑style
    measurement to extract features from the quantum state.
    """
    def __init__(self, num_features: int, kernel_size: int = 2):
        super().__init__()
        side = math.isqrt(num_features)
        self.kernel_size = kernel_size
        self.feature_dim = side - kernel_size + 1
        self.encoder = nn.Linear(num_features, side * side)
        self.quantum_filter = QuantumConvFilter(kernel_size)
        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Input of shape (batch, height, width).

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample.
        """
        batch_size = states.shape[0]
        # Flatten spatial dimensions for encoder
        flat = states.view(batch_size, -1)
        encoded = self.encoder(flat)
        features = []
        for i in range(batch_size):
            # Extract sliding windows
            sliding = encoded[i].reshape(-1, self.feature_dim)
            window_features = []
            for j in range(self.feature_dim):
                window = sliding[j].view(self.kernel_size, self.kernel_size).cpu().numpy()
                window_features.append(self.quantum_filter.run(window))
            features.append(window_features)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=states.device)
        return self.head(features_tensor).squeeze(-1)

__all__ = ["QRegressor", "RegressionDataset", "generate_superposition_data"]
