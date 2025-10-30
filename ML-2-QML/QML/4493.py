import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

# Configuration dataclass for the autoencoder
class AutoencoderConfig:
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64), dropout=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

# Simple fully‑connected autoencoder
class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# Wrapper for a parameterised quantum circuit
class QuantumCircuitWrapper:
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self._circuit)

        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# Differentiable bridge between PyTorch and the quantum circuit
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        expectation = ctx.circuit.run(thetas)
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.detach().cpu().numpy()):
            right = ctx.circuit.run([val + shift[idx]])[0]
            left = ctx.circuit.run([val - shift[idx]])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

# Hybrid head that forwards activations through the quantum circuit
class Hybrid(nn.Module):
    def __init__(self, n_qubits: int = 1, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

# Unified model that combines auto‑encoding, fully‑connected and quantum heads
class SharedClassName(nn.Module):
    def __init__(self, input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 n_qubits: int = 1,
                 shots: int = 100):
        super().__init__()
        config = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
        self.autoencoder = AutoencoderNet(config)
        self.fc = nn.Linear(latent_dim, 1)
        self.hybrid = Hybrid(n_qubits, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        logits = self.fc(z)
        probs = self.hybrid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["SharedClassName", "AutoencoderNet", "AutoencoderConfig", "HybridFunction"]
