"""
HybridClassifier – quantum‑augmented CNN with a variational circuit head.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile

class FeatureAttention(nn.Module):
    """Learnable attention over flattened features."""
    def __init__(self, in_features: int, hidden_size: int = 32):
        super().__init__()
        self.attn = nn.Linear(in_features, hidden_size)
        self.proj = nn.Linear(hidden_size, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = F.relu(self.attn(x))
        weights = F.softmax(self.proj(attn), dim=1)
        return x * weights

class VariationalQuantumCircuit:
    """Two‑qubit ansatz with parameter shift gradients."""
    def __init__(self, n_qubits: int, backend, shots: int, params_per_qubit: int = 3):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.params_per_qubit = params_per_qubit
        self.total_params = n_qubits * params_per_qubit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.ParameterVector('theta', self.total_params)
        # Simple RX‑RY‑RZ ansatz per qubit
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i * params_per_qubit], i)
            self.circuit.ry(self.theta[i * params_per_qubit + 1], i)
            self.circuit.rz(self.theta[i * params_per_qubit + 2], i)
        # Entangle
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter vectors.
        Returns expectation values of Z on qubit 0."""
        compiled = transpile(self.circuit, self.backend)
        results = []
        for theta in thetas:
            param_bind = dict(zip(self.theta, theta))
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_bind])
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                bit = int(bitstring[::-1][0])  # first qubit
                exp += (1 if bit == 0 else -1) * count
            exp /= self.shots
            results.append(exp)
        return np.array(results)

class QuantumExpectationFunction(torch.autograd.Function):
    """Differentiable wrapper around the variational circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        batch = inputs.detach().cpu().numpy()
        thetas = np.tile(batch[:, None], (1, circuit.total_params))
        exp = circuit.run(thetas)
        result = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        batch = inputs.detach().cpu().numpy()
        thetas_plus = np.tile((batch + shift)[:, None], (1, ctx.circuit.total_params))
        thetas_minus = np.tile((batch - shift)[:, None], (1, ctx.circuit.total_params))
        exp_plus = ctx.circuit.run(thetas_plus)
        exp_minus = ctx.circuit.run(thetas_minus)
        grad = 0.5 * (exp_plus - exp_minus)
        grad_tensor = torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a variational quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = VariationalQuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumExpectationFunction.apply(inputs, self.circuit, self.shift)

class HybridClassifier(nn.Module):
    """Hybrid CNN + quantum expectation head for binary classification."""
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
        self.attention = FeatureAttention(55815)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        x = self.hybrid(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs.unsqueeze(-1), 1 - probs.unsqueeze(-1)), dim=-1)

__all__ = ["VariationalQuantumCircuit", "QuantumExpectationFunction", "Hybrid", "HybridClassifier"]
