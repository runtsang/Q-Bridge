import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Parameterized two‑qubit circuit used in the hybrid head."""
    def __init__(self, backend, shots=1024):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = Parameter("theta")
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, 0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()
    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            bits = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(bits * probs)
        return np.array([expectation(result)])

class HybridFunctionQuantum(torch.autograd.Function):
    """Differentiable interface to the quantum expectation using a shift‑rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        thetas = inputs.detach().cpu().numpy().flatten()
        exp_vals = circuit.run(thetas)
        out = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy().flatten()
        grads = []
        for theta in thetas:
            exp_plus = ctx.circuit.run([theta + shift])[0]
            exp_minus = ctx.circuit.run([theta - shift])[0]
            grads.append(exp_plus - exp_minus)
        grad_input = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        return grad_input * grad_output, None, None

class HybridHeadQuantum(nn.Module):
    """Hybrid quantum head that outputs a two‑class probability vector."""
    def __init__(self, backend, shots=1024, shift=np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(backend, shots)
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        exp = HybridFunctionQuantum.apply(inputs, self.circuit, self.shift)
        probs = torch.sigmoid(exp)
        return torch.cat((probs, 1 - probs), dim=-1)

class HybridBinaryClassifierQML(nn.Module):
    """CNN backbone with a quantum hybrid head for binary classification."""
    def __init__(self, backend=AerSimulator(), shots=1024) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )
        self.head = HybridHeadQuantum(backend, shots)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)
        return self.head(x)
