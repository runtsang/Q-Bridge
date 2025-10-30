import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit as QC, assemble, transpile
from qiskit.providers.aer import AerSimulator

class ParametricAnsatz(QC):
    """Two‑qubit variational ansatz with Ry rotations and a CX gate."""
    def __init__(self, n_qubits: int = 2):
        super().__init__(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.theta = theta

        # Entangling block
        self.h(0)
        self.h(1)
        self.cx(0, 1)

        # Parameterised rotations
        self.ry(theta, 0)
        self.ry(theta, 1)

        # Final entanglement
        self.cx(0, 1)

        # No measurement – we will compute expectation from the statevector

class QuantumLayer:
    """Runs a parametric circuit on Aer and returns the expectation of Pauli‑Z."""
    def __init__(self, n_qubits: int = 2, shots: int = 512):
        self.backend = AerSimulator()
        self.circuit = ParametricAnsatz(n_qubits)
        self.shots = shots
        self.param = self.circuit.parameters[0]  # Only one parameter: theta

    def run(self, theta: np.ndarray) -> np.ndarray:
        """Return expectation value of Pauli‑Z on qubit 0 for each theta."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.param: t} for t in theta],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()

        exp = 0.0
        for bitstring, cnt in counts.items():
            # bitstring[0] corresponds to qubit 0 (most significant bit)
            val = 1.0 if bitstring[0] == "0" else -1.0
            exp += val * cnt
        exp /= self.shots
        return np.array([exp], dtype=np.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a scalar through the quantum layer."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_layer: QuantumLayer, shift: float):
        ctx.shift = shift
        ctx.quantum_layer = quantum_layer
        theta = inputs.detach().cpu().numpy().flatten()
        exp = ctx.quantum_layer.run(theta)
        out = torch.tensor(exp, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for val in inputs.detach().cpu().numpy().flatten():
            theta_plus = np.array([val + shift])
            theta_minus = np.array([val - shift])
            exp_plus = ctx.quantum_layer.run(theta_plus)
            exp_minus = ctx.quantum_layer.run(theta_minus)
            grad.append(exp_plus - exp_minus)
        grad = torch.tensor(grad, device=inputs.device, dtype=torch.float32)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a scalar through a quantum circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 512, shift: float = np.pi / 2):
        super().__init__()
        self.quantum_layer = QuantumLayer(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, 1)
        return HybridFunction.apply(inputs.squeeze(-1), self.quantum_layer, self.shift)

class QCNet(nn.Module):
    """CNN backbone followed by a quantum expectation head."""
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
        self.hybrid = Hybrid(n_qubits=2, shots=512, shift=np.pi / 2)

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
        x = self.fc3(x).squeeze(-1)   # (batch,)
        x = self.hybrid(x).view(-1, 1)  # (batch, 1)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ParametricAnsatz", "QuantumLayer", "HybridFunction", "Hybrid", "QCNet"]
