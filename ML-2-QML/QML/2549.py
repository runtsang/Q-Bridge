import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumSampler(nn.Module):
    """Parameterized 2‑qubit sampler returning a 4‑D probability vector."""
    def __init__(self, backend: qiskit.providers.BaseBackend) -> None:
        super().__init__()
        self.backend = backend
        self.qc = QuantumCircuit(2)
        inp = qiskit.circuit.ParameterVector('input', 2)
        wgt = qiskit.circuit.ParameterVector('weight', 4)
        self.qc.ry(inp[0], 0)
        self.qc.ry(inp[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(wgt[0], 0)
        self.qc.ry(wgt[1], 1)
        self.qc.cx(0, 1)
        self.qc.ry(wgt[2], 0)
        self.qc.ry(wgt[3], 1)
        self.qc.measure_all()
        self.inp_params = inp
        self.wgt_params = wgt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for inp in x:
            bind = {self.inp_params[0]: inp[0].item(),
                    self.inp_params[1]: inp[1].item()}
            compiled = transpile(self.qc, self.backend)
            qobj = assemble(compiled, parameter_binds=[bind], shots=1024)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            vec = np.zeros(4)
            for bitstr, cnt in counts.items():
                idx = int(bitstr, 2)
                vec[idx] = cnt
            vec = vec / 1024
            probs.append(vec)
        return torch.tensor(probs, dtype=torch.float)

class QuantumExpectation(nn.Module):
    """Differentiable quantum expectation layer using a 2‑qubit circuit."""
    def __init__(self, backend: qiskit.providers.BaseBackend) -> None:
        super().__init__()
        self.backend = backend
        self.circuit = QuantumCircuit(2)
        theta = qiskit.circuit.Parameter('theta')
        self.circuit.h([0, 1])
        self.circuit.ry(theta, 0)
        self.circuit.ry(theta, 1)
        self.circuit.measure_all()
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expectations = []
        for val in x.squeeze():
            bind = {self.theta: val.item()}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, parameter_binds=[bind], shots=1024)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            exp = 0.0
            for bitstr, cnt in counts.items():
                state = int(bitstr, 2)
                exp += state * cnt
            exp /= 1024
            expectations.append(exp)
        return torch.tensor(expectations, dtype=torch.float).unsqueeze(-1)

class HybridClassifier(nn.Module):
    """Convolutional backbone followed by a quantum expectation head."""
    def __init__(self, backend: qiskit.providers.BaseBackend) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.expectation = QuantumExpectation(backend)

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
        exp = self.expectation(x)
        probs = torch.sigmoid(exp)
        return torch.cat((probs, 1 - probs), dim=-1)

class UnifiedSamplerHybridNet(nn.Module):
    """Quantum‑classical hybrid sampler and classifier."""
    def __init__(self, backend: qiskit.providers.BaseBackend = AerSimulator()) -> None:
        super().__init__()
        self.sampler = QuantumSampler(backend)
        self.classifier = HybridClassifier(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.sampler(x)          # (batch, 4)
        img = sample.unsqueeze(1).repeat(1, 3, 1, 1)   # (batch, 3, 4, 4)
        img = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)
        return self.classifier(img)

__all__ = ["UnifiedSamplerHybridNet"]
