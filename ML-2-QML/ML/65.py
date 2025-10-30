import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qml_module import Hybrid

class DualHybridClassifier(nn.Module):
    """
    A binary classifier with a classical dense head and a quantum expectation head.
    The forward pass returns the classical probability, the quantum probability,
    and a joint probability that is a weighted average of the two.
    """
    def __init__(self,
                 in_features: int,
                 n_qubits: int = 2,
                 shift: float = np.pi / 2,
                 temperature: float = 1.0,
                 loss_weight: float = 0.5):
        super().__init__()
        # Classical head
        self.classical = nn.Linear(in_features, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # Quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots=200, shift=shift)
        # Weight for combining probabilities
        self.weight = nn.Parameter(torch.tensor([0.5]))
        self.loss_weight = loss_weight

    def forward(self, x: torch.Tensor):
        # Classical path
        logits = self.classical(x)
        probs_classical = torch.sigmoid(logits / self.temperature)
        # Quantum path
        logits_flat = logits.squeeze(-1)
        probs_quantum = self.hybrid(logits_flat)
        probs_quantum = torch.sigmoid(probs_quantum)
        # Joint probability
        probs_joint = self.weight * probs_classical + (1.0 - self.weight) * probs_quantum
        return torch.cat([probs_classical.unsqueeze(-1),
                          probs_quantum.unsqueeze(-1),
                          probs_joint.unsqueeze(-1)], dim=-1)
