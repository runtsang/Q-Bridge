import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile

class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers and optional 1x1 skip."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels!= out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        out += residual
        return F.relu(out)

class QuantumCircuitWrapper:
    """Two‑qubit parametric circuit executed on an Aer simulator."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, inputs: list[float]) -> list[float]:
        """Execute the circuit for each input value and return expectation of Z."""
        preds = []
        for val in inputs:
            compiled = transpile(self.circuit, self.backend)
            param_binds = [{self.theta: val}]
            qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            exp = 0.0
            for bitstring, count in result.items():
                z = 1 if bitstring == '0'*self.n_qubits else -1
                exp += z * count
            exp /= self.shots
            preds.append(exp)
        return preds

class HybridHead(nn.Module):
    """Hybrid head that fuses classical logits with a quantum expectation."""
    def __init__(self, in_features: int, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.quantum = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        q_exp = torch.tensor(
            self.quantum.run(logits.detach().cpu().numpy().flatten()),
            dtype=x.dtype,
            device=x.device
        ).unsqueeze(1)
        probs = torch.sigmoid(logits + self.shift * q_exp)
        return torch.cat((probs, 1 - probs), dim=-1)

class HybridBinaryClassifier(nn.Module):
    """CNN with residual blocks followed by a hybrid quantum‑classical head."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 200, shift: float = np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.res1 = ResidualBlock(6, 6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(15, 15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        if backend is None:
            backend = Aer.get_backend('aer_simulator')
        self.head = HybridHead(self.fc3.out_features, n_qubits, backend, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.pool(x)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.head(x)

__all__ = ["QuantumCircuitWrapper", "HybridHead", "HybridBinaryClassifier", "ResidualBlock"]
