import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import Aer, transpile, assemble
import qiskit

class QuantumCircuit:
    """Two‑qubit variational circuit for the hybrid head."""
    def __init__(self, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter('theta')
        self._circuit.h([0, 1])
        self._circuit.ry(self.theta, 0)
        self._circuit.cx(0, 1)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in result.keys()])
            probs = counts / self.shots
            z_vals = np.where(states & 1, -1, 1)
            return np.sum(z_vals * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the 2‑qubit circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        exp = circuit.run(angles)
        out = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.cpu().numpy(), ctx.shift)
        grads = []
        for inp, s in zip(inputs.cpu().numpy(), shift):
            exp_plus = ctx.circuit.run([inp + s])
            exp_minus = ctx.circuit.run([inp - s])
            grads.append(exp_plus - exp_minus)
        grad_inputs = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grad_inputs * grad_output.unsqueeze(1), None, None

class QuantumHybrid(nn.Module):
    """Wrapper that forwards activations through the 2‑qubit circuit."""
    def __init__(self, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that mimics a quantum patch."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class HybridQuanvolutionQCNet(nn.Module):
    """Hybrid network that uses a quantum circuit as the head."""
    def __init__(self):
        super().__init__()
        # Quanvolution filter
        self.quanv = QuanvolutionFilter()

        # Convolutional backbone
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected head
        self.fc1 = nn.Linear(15 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        backend = Aer.get_backend('aer_simulator')
        self.hybrid = QuantumHybrid(backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quanv(x)
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
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumCircuit", "QuantumHybridFunction",
           "QuantumHybrid", "QuanvolutionFilter",
           "HybridQuanvolutionQCNet"]
