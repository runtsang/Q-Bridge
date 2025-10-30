import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qml_code  # quantum module defined below

class HybridBinaryClassifier(nn.Module):
    """
    CNN‑based binary classifier with a quantum hybrid head and optional sampler.
    The architecture mirrors the original QCNet but replaces the quantum layer
    with a flexible, differentiable quantum expectation head.
    """
    def __init__(self,
                 num_qubits: int = 2,
                 quantum_depth: int = 2,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        # Classical feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)                     # output a single scalar
        )
        # Quantum hybrid head
        self.quantum_head = qml_code.HybridQuantumLayer(
            num_qubits=num_qubits, depth=quantum_depth, shift=shift
        )
        # Optional sampler network for probabilistic training
        self.sampler = qml_code.SamplerQNN(self.quantum_head.circuit.circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a two‑column probability distribution.
        """
        # Extract features → shape (batch, 1)
        features = self.feature_extractor(x)
        # Quantum expectation value → shape (batch, 1)
        q_val = self.quantum_head(features)
        # Convert to probabilities
        probs = torch.sigmoid(q_val)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
