"""Quantum implementation of the hybrid classifier with a variational circuit head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile, Aer

class FeatureSelector(nn.Module):
    """Selects the top‑k activations from a dense layer based on learned importance scores."""
    def __init__(self, in_features: int, k: int):
        super().__init__()
        self.k = k
        self.score = nn.Parameter(torch.randn(in_features))

    def forward(self, x: torch.Tensor):
        probs = F.softmax(self.score, dim=0)
        _, idx = torch.topk(probs, self.k)
        return torch.index_select(x, 1, idx)

class QuantumCircuitWrapper:
    """Parameterized quantum circuit with a configurable depth."""
    def __init__(self, n_qubits: int, depth: int, backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        for _ in range(depth):
            for q in range(n_qubits):
                self._circuit.ry(self.theta, q)
                self._circuit.rz(self.theta, q)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            exp_right = ctx.circuit.run([value + shift[idx]])
            exp_left = ctx.circuit.run([value - shift[idx]])
            gradients.append(exp_right - exp_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, depth: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, depth, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor):
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QuantumHybridClassifier(nn.Module):
    """CNN followed by a feature selector and a quantum expectation head."""
    def __init__(self, k: int = 10, depth: int = 2, n_qubits: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.selector = FeatureSelector(in_features=84, k=k)
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=n_qubits, depth=depth, backend=backend, shots=1024, shift=np.pi/2)

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
        x = self.selector(x)
        # Reduce selected features to a single scalar via mean
        x_mean = torch.mean(x, dim=1, keepdim=True)
        quantum_out = self.hybrid(x_mean)
        probs = torch.sigmoid(quantum_out)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
