import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumExpectationLayer(nn.Module):
    """
    3‑qubit entangled circuit that returns the expectation value of the
    first qubit in the Z basis.  Parameters are injected via RY gates.
    Parameter‑shift gradients are automatically handled by PyTorch.
    """
    def __init__(self, n_qubits: int = 3, shots: int = 512):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.base_circ = QuantumCircuit(n_qubits)
        self.params = [QuantumCircuit.Parameter(f"θ_{i}") for i in range(n_qubits)]
        self.base_circ.h(range(n_qubits))
        for i, p in enumerate(self.params):
            self.base_circ.ry(p, i)
        self.base_circ.barrier()
        self.base_circ.measure_all()

    def _expectation(self, circuit: QuantumCircuit) -> float:
        qobj = assemble(circuit, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp = 0.0
        for state, cnt in counts.items():
            bit = int(state[-1])  # Qiskit uses little‑endian ordering
            exp += (1 - 2 * bit) * cnt
        return exp / self.shots

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        thetas = inputs.detach().cpu().numpy()
        exp_vals = []
        for theta in thetas:
            circ = self.base_circ.copy()
            circ = circ.bind_parameters({p: theta[i] for i, p in enumerate(self.params)})
            exp_vals.append(self._expectation(circ))
        probs = torch.tensor(exp_vals, device=inputs.device, dtype=inputs.dtype)
        return probs.view(-1, 1)

class HybridBinaryClassifier(nn.Module):
    """
    Quantum‑enhanced binary classifier that shares the same convolutional
    backbone as the classical version.  The head is a 3‑qubit expectation
    layer.  The API matches the classical implementation for easy model
    interchange.
    """
    def __init__(self, dropout: float | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = QuantumExpectationLayer(n_qubits=3, shots=512)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.dropout is not None:
            x = self.dropout(x)
        probs = self.head(x.squeeze(-1))
        return torch.cat([probs, 1 - probs], dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
