import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Parametrised ansatz with 3 qubits and 2 layers."""
    def __init__(self, n_qubits: int = 3, backend=None, shots: int = 2048):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # Layer 1
        self.circuit.h(range(n_qubits))
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)
        # Parametrised rotations
        for q in range(n_qubits):
            self.circuit.ry(self.theta, q)
        # Layer 2
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()], dtype=float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float, scale: float):
        ctx.shift = shift
        ctx.scale = scale
        ctx.circuit = circuit
        # inputs: [batch]
        thetas = inputs.detach().cpu().numpy().flatten()
        expectations = circuit.run(thetas)
        probs = torch.sigmoid(torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device) + shift)
        ctx.save_for_backward(inputs, probs)
        return probs * scale

    @staticmethod
    def backward(ctx, grad_output):
        inputs, probs = ctx.saved_tensors
        shift = ctx.shift
        scale = ctx.scale
        circuit = ctx.circuit
        eps = 1e-3
        grad_expectations = []
        for val in inputs.detach().cpu().numpy().flatten():
            exp_plus = circuit.run([val + eps])[0]
            exp_minus = circuit.run([val - eps])[0]
            grad_expectations.append((exp_plus - exp_minus) / (2 * eps))
        grad_expectations = torch.tensor(grad_expectations, dtype=inputs.dtype, device=inputs.device)
        sigmoid_deriv = probs * (1 - probs)
        grad_inputs = grad_output * scale * sigmoid_deriv * grad_expectations
        return grad_inputs, None, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards through a quantum circuit."""
    def __init__(self, n_qubits: int = 3, backend=None, shots: int = 2048, shift: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift
        self.scale = scale

    def forward(self, inputs: torch.Tensor):
        # inputs shape [batch, 1] or [batch]
        squeezed = torch.squeeze(inputs) if inputs.shape[-1] == 1 else inputs
        return HybridFunction.apply(squeezed, self.circuit, self.shift, self.scale)

class QCNet(nn.Module):
    """CNN followed by a quantum expectation head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=3, backend=backend, shots=2048,
                             shift=np.pi / 2, scale=1.0)

    def forward(self, inputs: torch.Tensor):
        x = F.relu(self.conv1(inputs))
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
        probs = self.hybrid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
