import qiskit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from qiskit import assemble, transpile
from qiskit import Aer

class QuantumCircuitWrapper:
    """Parameterized quantum circuit that returns expectation values."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        all_qubits = list(range(n_qubits))
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
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
            states = np.array([int(k, 2) for k in count_dict.keys()])
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
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        gradients = []
        for idx, val in enumerate(inputs.numpy()):
            exp_right = ctx.circuit.run([val + shift[idx]])
            exp_left = ctx.circuit.run([val - shift[idx]])
            gradients.append(exp_right - exp_left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QuantumFilterWrapper(nn.Module):
    """Quantum filter that replaces the classical convolutional filter."""
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
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
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

class ConvHybridNet(nn.Module):
    """Hybrid convolutional network that uses a quantum filter and quantum head."""
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 kernel_size: int = 3,
                 use_quantum_filter: bool = True,
                 use_quantum_head: bool = True,
                 device: str = 'cpu'):
        super().__init__()
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
        self.filter = QuantumFilterWrapper(kernel_size=kernel_size,
                                           backend=qiskit.Aer.get_backend("qasm_simulator"),
                                           shots=100,
                                           threshold=127)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.head = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.filter(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.head(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["ConvHybridNet", "QuantumFilterWrapper", "Hybrid", "HybridFunction", "QuantumCircuitWrapper"]
