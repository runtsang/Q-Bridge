import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import Aer
from qiskit.providers.aer.noise import NoiseModel

class QuantumCircuit:
    """Parameterized variational circuit with adjustable depth."""
    def __init__(self, n_qubits: int, depth: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits * depth)]

        # Build a simple layered ansatz: RY on each qubit followed by CNOT chain
        for d in range(depth):
            for q in range(n_qubits):
                self.circuit.ry(self.params[d * n_qubits + q], q)
            for q in range(n_qubits - 1):
                self.circuit.cx(q, q + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter sets."""
        # thetas shape: (batch, n_qubits * depth)
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: t for p, t in zip(self.params, theta)}
                             for theta in thetas],
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
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # inputs shape: (batch,)
        expectation = circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        gradients = []
        for val, s in zip(inputs.detach().cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])
            left = ctx.circuit.run([val - s])
            gradients.append(right - left)
        grad_inputs = torch.tensor(gradients, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, depth: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, depth, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs shape: (batch,)
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self, use_noise: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if use_noise:
            noise_model = NoiseModel.from_backend(Aer.get_backend("aer_simulator"))
            backend = Aer.get_backend("aer_simulator", noise_model=noise_model)
        else:
            backend = Aer.get_backend("aer_simulator")

        self.hybrid = Hybrid(
            n_qubits=2,
            depth=3,
            backend=backend,
            shots=200,
            shift=np.pi / 2
        )

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
        # Quantum expectation head expects a 1‑D tensor
        x = self.hybrid(x.squeeze(-1))
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
