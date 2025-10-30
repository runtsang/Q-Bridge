import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.providers import BaseBackend
from qiskit.circuit import Parameter

class QuantumKernelCircuit(nn.Module):
    """Parametric quantum circuit that encodes a scalar feature and a fixed reference angle, returning a Z‑expectation value."""
    def __init__(self,
                 backend: BaseBackend = AerSimulator(),
                 shots: int = 1024,
                 ref_angle: float = np.pi / 4) -> None:
        super().__init__()
        self.backend = backend
        self.shots = shots
        self.ref_angle = ref_angle

        self.theta = Parameter("θ")
        self.circuit = qiskit.QuantumCircuit(2)
        # Encode the input feature on qubit 0
        self.circuit.ry(self.theta, 0)
        # Encode a fixed reference on qubit 1
        self.circuit.ry(self.ref_angle, 1)
        # Entangle the qubits
        self.circuit.cx(0, 1)
        # Measure all qubits
        self.circuit.measure_all()

    def _run_single(self, theta: float) -> float:
        """Execute the circuit for a single input value and return the Z‑expectation on qubit 0."""
        compiled = transpile(self.circuit, self.backend)
        bound = compiled.bind_parameters({self.theta: theta})
        qobj = assemble(bound, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp_val = 0.0
        for bitstring, count in counts.items():
            # bitstring[0] corresponds to qubit 0
            z = 1.0 if bitstring[0] == "0" else -1.0
            exp_val += z * count
        return exp_val / self.shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch processing of a tensor of shape (batch, 1)."""
        if x.ndim!= 2 or x.size(1)!= 1:
            raise ValueError("Input must be of shape (batch, 1)")
        results = [self._run_single(val.item()) for val in x]
        return torch.tensor(results, dtype=torch.float32, device=x.device).unsqueeze(1)

class HybridBinaryClassifier(nn.Module):
    """CNN backbone followed by a quantum kernel head for binary classification."""
    def __init__(self,
                 backend: BaseBackend = AerSimulator(),
                 shots: int = 1024,
                 ref_angle: float = np.pi / 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_head = QuantumKernelCircuit(backend, shots, ref_angle)

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
        # quantum head expects a scalar per sample
        x = x.view(x.size(0), 1)
        logits = self.quantum_head(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier", "QuantumKernelCircuit"]
