import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import Aer, transpile, assemble
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.aer import AerSimulator

class QuantumExpectationLayer(nn.Module):
    """Quantum layer that evaluates expectation of Z on a single qubit."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = AerSimulator()
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        theta = Parameter('θ')
        qc.h(list(range(self.n_qubits)))
        qc.ry(theta, list(range(self.n_qubits)))
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thetas = x.detach().cpu().numpy().flatten()
        compiled = transpile(self.circuit, self.backend)
        job = self.backend.run(assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[0]: theta} for theta in thetas]
        ))
        result = job.result().get_counts()
        if isinstance(result, list):
            expectations = np.array([self._expectation(c) for c in result])
        else:
            expectations = np.array([self._expectation(result)])
        return torch.tensor(expectations, dtype=torch.float32, device=x.device).unsqueeze(-1)

    def _expectation(self, count_dict):
        counts = np.array(list(count_dict.values()))
        states = np.array(list(count_dict.keys()), dtype=float)
        probs = counts / self.shots
        return np.dot(states, probs)

class HybridQCNet(nn.Module):
    """CNN with a quantum expectation head for multi‑class classification."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_head = QuantumExpectationLayer(n_qubits=1, shots=200)
        self.classifier = nn.Linear(1, num_classes)

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
        q_out = self.quantum_head(x)
        logits = self.classifier(q_out)
        return logits

__all__ = ["QuantumExpectationLayer", "HybridQCNet"]
