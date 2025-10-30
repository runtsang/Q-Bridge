import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit import ParameterVector

class QuantumCircuitWrapper:
    """Parameterized circuit with a random layer and an ansatz."""
    def __init__(self, n_qubits: int, depth: int, backend=None, shots=1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.theta = ParameterVector("theta", n_qubits * depth)
        self.circuit = self._build()

    def _build(self):
        qc = QuantumCircuit(self.n_qubits)
        # Random layer (mimics QFCModel RandomLayer)
        for q in range(self.n_qubits):
            qc.h(q)
            qc.rz(np.random.randn(), q)
        # Parameterised ansatz
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                qc.ry(self.theta[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                qc.cz(q, q+1)
        qc.measure_all()
        return qc

    def run(self, params: np.ndarray):
        param_bind = {self.theta[i]: params[i] for i in range(len(self.theta))}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, parameter_binds=[param_bind], shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp = 0.0
        for state, cnt in counts.items():
            z = np.array([1 if bit == '0' else -1 for bit in state[::-1]])
            exp += np.sum(z) * cnt
        exp /= self.shots
        return np.array([exp])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        params = inputs.cpu().detach().numpy()
        out = np.array([circuit.run(p)[0] for p in params])
        return torch.tensor(out, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs, = ctx.saved_tensors
        grads = []
        for p in inputs:
            grad_p = []
            for j in range(len(p)):
                p_plus = p.clone()
                p_plus[j] += shift
                p_minus = p.clone()
                p_minus[j] -= shift
                grad_j = circuit.run(p_plus.cpu().numpy())[0] - circuit.run(p_minus.cpu().numpy())[0]
                grad_p.append(grad_j)
            grads.append(grad_p)
        grad_tensor = torch.tensor(grads, dtype=torch.float32)
        return grad_tensor * grad_output.unsqueeze(-1), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, depth: int, shift: float = np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, depth)
        self.shift = shift

    def forward(self, inputs: torch.Tensor):
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QuantumClassifierModel(nn.Module):
    """CNN backbone followed by a quantum‑expectation head."""
    def __init__(self, n_qubits: int = 4, depth: int = 2, shift: float = np.pi/2):
        super().__init__()
        # CNN backbone identical to the ML version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = Hybrid(n_qubits, depth, shift)

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

        q_out = self.hybrid(x)
        probs = torch.softmax(q_out, dim=-1)
        return probs

__all__ = ["QuantumClassifierModel", "Hybrid", "HybridFunction", "QuantumCircuitWrapper"]
