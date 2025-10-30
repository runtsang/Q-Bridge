import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import numpy as np
from typing import Callable, Iterable, List

class QuantumCircuitWrapper:
    def __init__(self, n_qubits: int, backend, shots: int):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: th} for th in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.numpy())
        out = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for val, sh in zip(inputs.numpy(), shift):
            right = ctx.circuit.run([val + sh])
            left = ctx.circuit.run([val - sh])
            grads.append(right - left)
        grad = torch.tensor(grads, dtype=torch.float32)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum head that maps a scalar through a small NN to circuit angles."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift
        self.param_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.param_net(x.unsqueeze(1)).squeeze(1)
        return HybridFunction.apply(angles, self.circuit, self.shift)

class HybridQCNet(nn.Module):
    """Hybrid convolutional classifier with a quantum expectation head."""
    def __init__(self, use_quantum: bool = True):
        super().__init__()
        self.use_quantum = use_quantum
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        if self.use_quantum:
            backend = qiskit.Aer.get_backend("aer_simulator")
            self.head = Hybrid(1, backend, shots=100, shift=np.pi/2)
        else:
            self.head = nn.Sigmoid()

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
        logits = self.head(x)
        return torch.cat((logits, 1 - logits), dim=-1)

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: List[List[float]]) -> List[List[float]]:
        """Evaluate the model for a list of parameter sets and observables."""
        results = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                out = self.forward(torch.tensor(params).unsqueeze(0))
                row = []
                for obs in observables:
                    val = obs(out)
                    row.append(float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results
