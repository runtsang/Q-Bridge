"""Quantum‑only version of the hybrid model using Pennylane.

The `QFCModelHybrid` class implements the same variational circuit as
the TorchQuantum version but targets a Pennylane device.  It accepts
classical image features, prepares a state vector via a linear layer,
and runs a parameter‑reversible circuit with RX/RZ rotations and CX
entanglement.  The output is a four‑dimensional feature vector ready
for classification.
"""

import pennylane as qml
import torch
import torch.nn as nn

class QFCModelHybrid(nn.Module):
    """Hybrid classical‑quantum model using Pennylane."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        # Classical feature extractor (same as PyTorch version)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Linear bottleneck to match quantum dimension
        self.bottleneck = nn.Linear(16 * 28 * 28, 4 * n_wires)
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Learnable parameters for each variational layer
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))
        # Pennylane device (default qubit simulator)
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.norm = nn.BatchNorm1d(n_wires)

    def _quantum_circuit(self, params):
        """Variational circuit applied to the state prepared by the linear layer."""
        qml.Hadamard(wires=range(self.n_wires))
        for i in range(self.n_layers):
            for w in range(self.n_wires):
                qml.RX(params[i, w, 0], wires=w)
                qml.RY(params[i, w, 1], wires=w)
                qml.RZ(params[i, w, 2], wires=w)
            # Entangle adjacent qubits
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        prep = self.bottleneck(flat)
        # Prepare a qnode
        @qml.qnode(self.dev, interface="torch")
        def circuit(params):
            return self._quantum_circuit(params)
        # Run the circuit for each sample in the batch
        out = []
        for i in range(bsz):
            out.append(circuit(self.params))
        out = torch.stack(out)
        return self.norm(out)
