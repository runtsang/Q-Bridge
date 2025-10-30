import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer simulator."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> float:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas]
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=float)
        return np.sum(states * probs)

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that feeds a scalar through the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        expectation = circuit.run(inputs.tolist())
        output = torch.tensor([expectation])
        ctx.save_for_backward(inputs, output)
        ctx.circuit = circuit
        ctx.shift = shift
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for v in inputs:
            right = ctx.circuit.run([v + shift])
            left = ctx.circuit.run([v - shift])
            grads.append(right - left)
        grads = torch.tensor(grads).float()
        return grads * grad_output.float(), None, None

class AutoencoderNet(nn.Module):
    """Same autoencoder structure as in the classical module."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class KernelQuantum(tq.QuantumModule):
    """TorchQuantum implementation of a simple quantum kernel."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.RY(n_wires=self.n_wires)  # placeholder for the encoding

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        for i, val in enumerate(x[0]):
            self.q_device.apply("ry", wires=[i], params=[val])
        for i, val in enumerate(y[0]):
            self.q_device.apply("ry", wires=[i], params=[-val])
        return torch.abs(self.q_device.states.view(-1)[0])

class HybridAutoEncoderKernelQCNet(nn.Module):
    """Hybrid quantum‑classical model mirroring the classical architecture."""
    def __init__(self,
                 in_channels: int = 3,
                 num_qubits: int = 4,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        backend = AerSimulator()
        self.quantum_circuit = QuantumCircuit(num_qubits, backend, shots=200)
        self.shift = np.pi / 2

        # Quantum kernel
        self.kernel = KernelQuantum()

        # Autoencoder for reconstruction regularisation
        self.autoencoder = AutoencoderNet(1, latent_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        quantum_out = HybridFunction.apply(x, self.quantum_circuit, self.shift)
        probs = torch.cat([quantum_out, 1 - quantum_out], dim=-1)

        recon = self.autoencoder(x)
        return probs, recon

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix via the quantum kernel."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridAutoEncoderKernelQCNet"]
