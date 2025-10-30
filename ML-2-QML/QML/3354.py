"""Hybrid quantum‑classical classifier that compresses CNN features
with a classical autoencoder before feeding them to a parameterised
quantum circuit.  The quantum head replaces the final dense layer
of the purely classical model, enabling a scalable hybrid approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
from.ml_module import Autoencoder

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "HybridAutoencoderQCNet"]

class QuantumCircuit:
    """Two‑qubit parameterised circuit executed on AerSimulator."""
    def __init__(self, n_qubits: int, shots: int = 256) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles and return expectation."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch tensors and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        theta_vals = inputs.cpu().numpy().flatten()
        exp_vals = circuit.run(theta_vals)
        out = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for i, val in enumerate(inputs.cpu().numpy()):
            right = ctx.circuit.run([val + shift[i]])[0]
            left = ctx.circuit.run([val - shift[i]])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, shift: float = np.pi / 2, shots: int = 256) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(flat, self.circuit, self.shift)

class HybridAutoencoderQCNet(nn.Module):
    """CNN → autoencoder → quantum hybrid classifier."""
    def __init__(
        self,
        n_classes: int = 2,
        autoencoder_latent: int = 2,
        quantum_qubits: int = 2,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Autoencoder feature compressor
        self.autoencoder = Autoencoder(1, latent_dim=autoencoder_latent)

        # Quantum hybrid head
        self.hybrid = Hybrid(quantum_qubits, shift=np.pi / 2, shots=256)

        self.n_classes = n_classes

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
        x = self.fc3(x)  # shape (batch, 1)

        # Compress with autoencoder
        latent = self.autoencoder.encode(x)

        # Quantum hybrid classification
        quantum_out = self.hybrid(latent)

        probs = torch.sigmoid(quantum_out).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)
