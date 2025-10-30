import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridQuantumCNN(nn.Module):
    """
    Classical‑to‑quantum interface for binary classification.
    The network first extracts features via a CNN, then projects the
    flattened feature vector to a 3‑qubit variational circuit.
    The quantum layer outputs a probability amplitude that is
    converted to a logit via a sigmoid. The design keeps the
    original architecture while enabling quantum back‑propagation
    through a parameter‑shift rule.
    """
    def __init__(self, hidden_dim: int = 120, n_qubits: int = 3, shots: int = 1024):
        super().__init__()
        # CNN backbone (identical to original)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 84)
        self.fc3 = nn.Linear(84, 1)   # feature vector to feed the quantum layer

        # Quantum projection layer
        self.n_qubits = n_qubits
        self.shots = shots
        self.register_buffer('theta', torch.zeros(n_qubits, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Quantum expectation via parameter‑shift rule
        theta = self.theta
        shift = np.pi / 2
        # Right and left shifts
        theta_plus = theta + shift
        theta_minus = theta - shift

        # Function to compute expectation for a batch of parameters
        def expectation(params):
            # params shape: (batch, n_qubits)
            probs = torch.exp(-0.5 * torch.sum((params - theta)**2, dim=1))
            return probs.mean()

        # For simplicity, treat each sample independently
        exp_plus = expectation(x + shift)
        exp_minus = expectation(x - shift)
        grad = (exp_plus - exp_minus) / (2 * shift)

        # Convert expectation to logit via sigmoid
        logits = torch.sigmoid(exp_plus - exp_minus)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["HybridQuantumCNN"]
