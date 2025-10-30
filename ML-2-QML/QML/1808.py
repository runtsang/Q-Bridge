import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class VariationalQuantumCircuit:
    """Two‑qubit parameterised circuit used as a quantum head."""
    def __init__(self, n_qubits: int = 2, shots: int = 200, backend=None):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend if backend is not None else AerSimulator()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("θ")
        # Simple entangling pattern
        for q in range(n_qubits):
            self.circuit.ry(self.theta, q)
            if q < n_qubits - 1:
                self.circuit.cx(q, q + 1)
        self.circuit.measure_all()

    def run(self, param: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: float(param)}])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of Z on the first qubit
        exp = 0.0
        for bitstring, cnt in result.items():
            z = 1 if bitstring[-1] == '0' else -1
            exp += z * cnt
        return np.array([exp / self.shots])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the variational circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        # Shift inputs
        shifted = inputs + shift
        expectation = circuit.run(shifted.detach().cpu().numpy())
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = ctx.circuit.run(val + shift)[0]
            exp_minus = ctx.circuit.run(val - shift)[0]
            grad = (exp_plus - exp_minus) / 2.0
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
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
        self.quantum_circuit = VariationalQuantumCircuit(n_qubits=2, shots=200)
        self.shift = np.pi / 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        probs = HybridFunction.apply(x.squeeze(), self.quantum_circuit, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["VariationalQuantumCircuit", "HybridFunction", "QCNet"]
