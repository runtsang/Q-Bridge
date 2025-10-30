"""Hybrid classical-quantum estimator combining a classical quanvolution filter with a simulated quantum kernel for regression.

The module defines a single class `HybridEstimatorQNN` that:
- extracts 2×2 patches from a 28×28 grayscale image,
- encodes each patch into a 4‑qubit state via Ry rotations,
- applies a fixed entangling layer (CNOT chain),
- measures the expectation of Pauli‑Z on each qubit,
- flattens all patch measurements into a feature vector,
- feeds the vector into a linear regressor.
"""

import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer import AerSimulator

class QuanvolutionFilter(nn.Module):
    """Classical filter that simulates a 4‑qubit quantum kernel on image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.backend = AerSimulator(method="statevector")

    def _create_circuit(self, patch: torch.Tensor) -> QuantumCircuit:
        """Build a 4‑qubit circuit for a single patch."""
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(patch):
            qc.ry(val.item() * torch.pi.item() / 2, i)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        return qc

    def _expectation_z(self, result) -> list[float]:
        """Compute expectation of Pauli‑Z on each qubit from measurement counts."""
        counts = result.get_counts()
        total = sum(counts.values())
        exp_z = []
        for q in range(self.n_wires):
            exp = 0.0
            for bitstring, cnt in counts.items():
                bit = int(bitstring[self.n_wires - 1 - q])
                exp += cnt * (-1) ** bit
            exp_z.append(exp / total)
        return exp_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum kernel to all 2×2 patches of the input image."""
        bsz = x.shape[0]
        features = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].reshape(bsz, -1)
                patch_feats = []
                for i in range(bsz):
                    qc = self._create_circuit(patch[i])
                    result = execute(qc, self.backend, shots=1024).result()
                    patch_feats.append(self._expectation_z(result))
                patch_feats = torch.tensor(patch_feats, dtype=torch.float32, device=x.device)
                features.append(patch_feats)
        return torch.cat(features, dim=1)

class HybridEstimatorQNN(nn.Module):
    """Hybrid regressor that combines the quanvolution filter with a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        return self.linear(features)

__all__ = ["HybridEstimatorQNN"]
