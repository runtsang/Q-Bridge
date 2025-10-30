import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """Simple variational ansatz with Ry rotations and a CNOT chain."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.params = [self.circuit.placeholder(f'theta_{i}') for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.ry(self.params[i], i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, param_vals: np.ndarray) -> float:
        bind_dict = {self.params[i]: param_vals[i] for i in range(self.n_qubits)}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind_dict])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Expectation of Z on the first qubit
        expectation = 0.0
        for state, cnt in counts.items():
            z = 1 if state[-1] == '0' else -1
            expectation += z * cnt
        expectation /= self.shots
        return expectation

class QuantumFunction(torch.autograd.Function):
    """Differentiable interface that forwards gradients to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, qc: VariationalQuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.qc = qc
        exps = []
        for inp in inputs:
            exp = qc.run(inp.tolist())
            exps.append(exp)
        out = torch.tensor(exps, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for inp in inputs:
            right = ctx.qc.run((inp + shift).tolist())
            left = ctx.qc.run((inp - shift).tolist())
            grad = (right - left) / (2 * shift)
            grads.append(grad)
        grad_tensor = torch.tensor(grads, device=inputs.device, dtype=torch.float32)
        return grad_tensor * grad_output, None, None

class HybridQuantumHead(nn.Module):
    """Quantum head that maps a 4‑dimensional feature vector to a scalar."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = 0.1):
        super().__init__()
        self.qc = VariationalQuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumFunction.apply(x, self.qc, self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """Convolutional backbone followed by a variational quantum head."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Linear layer that outputs 4 features for the 4‑qubit encoding
        self.fc = nn.Linear(32, 4)
        backend = AerSimulator()
        self.quantum_head = HybridQuantumHead(4, backend, shots=1024, shift=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        logits = self.quantum_head(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["VariationalQuantumCircuit", "QuantumFunction",
           "HybridQuantumHead", "HybridQuantumBinaryClassifier"]
