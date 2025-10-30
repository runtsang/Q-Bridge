import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalHead(nn.Module):
    """Simple linear head used when the quantum option is off."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class QuantumHead(nn.Module):
    """
    Wrapper that forwards activations through a quantum circuit.
    The circuit must expose a ``run`` method that accepts a list of
    parameters and returns a NumPy array containing the expectation value.
    """
    def __init__(self, in_features: int, circuit, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.shift = shift
        self.circuit = circuit
        # Linear layer that maps the CNN output to a single parameter
        # for the quantum circuit.
        self.param_mapper = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to a single quantum parameter
        theta = self.param_mapper(x).squeeze(-1).cpu().numpy()
        # Execute the circuit
        exp_val = self.circuit.run(theta)
        # Convert back to a tensor on the original device
        return torch.tensor(exp_val, dtype=x.dtype, device=x.device)


class HybridClassifier(nn.Module):
    """
    CNN backbone followed by an interchangeable head.
    ``use_quantum`` selects between a classical or quantum head.
    """
    def __init__(self, use_quantum: bool = False, circuit=None, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.use_quantum = use_quantum

        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Head selection
        if self.use_quantum:
            if circuit is None:
                raise ValueError("Quantum circuit must be provided when use_quantum=True")
            self.head = QuantumHead(self.fc3.out_features, circuit, shift)
        else:
            self.head = ClassicalHead(self.fc3.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.head(x)
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["HybridClassifier"]
