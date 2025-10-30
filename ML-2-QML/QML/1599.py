import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit as QiskitCircuit, Parameter

class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit returning the expectation of Z on the first qubit."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = QiskitCircuit(n_qubits)
        self.theta = Parameter("θ")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        bind_list = [{self.theta: float(theta)} for theta in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=bind_list)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        def expectation(count_dict):
            probs = {}
            total = self.shots
            for bitstring, cnt in count_dict.items():
                prob = cnt / total
                z = 1 if bitstring[-1] == "0" else -1
                probs[z] = probs.get(z, 0) + prob
            return probs.get(1, 0) - probs.get(-1, 0)
        if isinstance(counts, list):
            return np.array([expectation(ct) for ct in counts])
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a scalar input to the quantum circuit."""
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        theta = input_tensor.detach().cpu().numpy()
        exp_val = circuit.run(theta)[0]
        out = torch.tensor([exp_val], dtype=input_tensor.dtype, device=input_tensor.device)
        ctx.save_for_backward(input_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        shift = ctx.shift
        theta = input_tensor.detach().cpu().numpy()
        grads = []
        for val in theta:
            right = ctx.circuit.run(np.array([val + shift]))[0]
            left = ctx.circuit.run(np.array([val - shift]))[0]
            grads.append((right - left) / 2)
        grad_input = torch.tensor(grads, dtype=input_tensor.dtype, device=input_tensor.device)
        return grad_input * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer mapping a scalar to a quantum expectation using parameter‑shift."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x.squeeze(), self.circuit, self.shift)

class QuantumHybridClassifier(nn.Module):
    """CNN followed by a quantum expectation head, mirroring the classical backbone."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits, backend, shots, shift=np.pi / 2)

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
        x = self.hybrid(x).t()
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "QuantumHybridClassifier"]
