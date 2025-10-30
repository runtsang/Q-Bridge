import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridQuantumNAT(nn.Module):
    """Classical CNN + optional quantum‑style fully‑connected layer.

    The architecture mirrors the classical seed (CNN → FC → BN) but
    exposes a ``use_quantum_fc`` flag that, when True, replaces the
    linear projection with a deterministic surrogate that mimics a
    quantum expectation.  This surrogate is kept purely classical
    (no Qiskit or torchquantum imports) so that the module can be
    used in any conventional PyTorch training loop.  The quantum
    version is provided separately in ``qml_code``."""
    def __init__(self, use_quantum_fc: bool = False, n_qubits: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc_proj = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU())
        self.use_quantum_fc = use_quantum_fc
        if use_quantum_fc:
            self.quantum_fc = ClassicalQuantumSurrogate(n_qubits=n_qubits)
        else:
            self.linear_fc = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        proj = self.fc_proj(flat)
        if self.use_quantum_fc:
            out = self.quantum_fc(proj)
        else:
            out = self.linear_fc(proj)
        return self.norm(out)

class ClassicalQuantumSurrogate(nn.Module):
    """Deterministic surrogate that emulates a quantum expectation.

    The surrogate takes a 64‑dim vector, truncates it to ``n_qubits``
    components, applies a non‑linear squashing, and returns a 4‑dim
    vector.  It is intentionally lightweight so that the classical
    training pipeline remains unchanged."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.proj = nn.Linear(n_qubits, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce dimensionality and squash
        angles = torch.tanh(x[:, :self.n_qubits])
        # Return a 4‑dim output via a simple linear mapping
        return self.proj(angles)
__all__ = ["HybridQuantumNAT", "ClassicalQuantumSurrogate"]
