import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter that emulates the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])

class QuantumFilterWrapper(nn.Module):
    """Wraps a parameterised quantum circuit as a PyTorch module."""
    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 127):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expected shape: (batch, 1, H, W)
        batch = data.shape[0]
        results = []
        for i in range(batch):
            flat = data[i, 0].reshape(-1).detach().cpu().numpy()
            bind = {theta: (np.pi if val > self.threshold else 0)
                    for theta, val in zip(self.theta, flat)}
            job = qiskit.execute(self.circuit,
                                 self.backend,
                                 shots=self.shots,
                                 parameter_binds=[bind])
            result = job.result().get_counts(self.circuit)
            total = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                total += ones * val
            results.append(total / (self.shots * self.n_qubits))
        return torch.tensor(results, dtype=torch.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that can be swapped with a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that can be replaced by a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridFunction.apply(logits, self.shift)

class ConvHybridNet(nn.Module):
    """Hybrid convolutional network that can swap classical/quantum filters and heads."""
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 kernel_size: int = 3,
                 use_quantum_filter: bool = False,
                 use_quantum_head: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.use_quantum_filter = use_quantum_filter
        self.use_quantum_head = use_quantum_head
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Filter
        if use_quantum_filter:
            self.filter = QuantumFilterWrapper(kernel_size=kernel_size,
                                               backend=qiskit.Aer.get_backend("qasm_simulator"),
                                               shots=100,
                                               threshold=127)
        else:
            self.filter = ConvFilter(kernel_size=kernel_size, threshold=0.0)
        # Head
        if use_quantum_head:
            self.head = Hybrid(self.fc3.out_features, shift=0.0)
        else:
            self.head = Hybrid(self.fc3.out_features, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.filter(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ConvHybridNet", "ConvFilter", "QuantumFilterWrapper", "Hybrid", "HybridFunction"]
