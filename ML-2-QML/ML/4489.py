import torch
import torch.nn as nn
import numpy as np
from.qml_module import HybridQuantumCircuit

class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected layer that can operate in purely classical mode
    or hybrid classical‑quantum mode.  The quantum sub‑module is a
    parameterized circuit built from QCNN‑style convolution and pooling
    layers (see :mod:`qml_module`).
    """

    def __init__(self, n_features: int = 1, n_qubits: int = 4, use_quantum: bool = True):
        super().__init__()
        self.use_quantum = use_quantum
        self.classical = nn.Linear(n_features, 1)
        self.quantum = HybridQuantumCircuit(n_qubits) if use_quantum else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  If ``use_quantum`` is True, the input is fed to the
        quantum circuit and the expectation value is returned; otherwise
        a simple linear transform is applied.
        """
        if self.use_quantum and self.quantum is not None:
            # Convert to numpy for the quantum backend
            thetas = x.detach().cpu().numpy().flatten()
            q_out = self.quantum.run(thetas)
            return torch.tensor(q_out, dtype=x.dtype, device=x.device)
        else:
            return self.classical(x)

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int):
        """
        Utility that mirrors :func:`generate_superposition_data` from
        ``QuantumRegression``.  Generates inputs and labels for a
        sinusoidal target function.
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary containing the input state and
    target value.  Mirrors the class from ``QuantumRegression``.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = HybridFCL.generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["HybridFCL", "RegressionDataset"]
