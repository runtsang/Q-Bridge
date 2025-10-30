import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderSettings:
    """Configuration for the simple feed‑forward autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class EncoderAutoNet(nn.Module):
    """Lightweight fully‑connected autoencoder used as a latent encoder."""
    def __init__(self, cfg: AutoencoderSettings) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def build_autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> EncoderAutoNet:
    cfg = AutoencoderSettings(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return EncoderAutoNet(cfg)

class QuantumCircuitWrapper:
    """Two‑qubit parameterised circuit executed on the Aer simulator."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.theta = theta
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys())).astype(float)
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class QuantumHybridLayer(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inp: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        exp = ctx.circuit.run(inp.tolist())
        out = torch.tensor([exp])
        ctx.save_for_backward(inp, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        inp, _ = ctx.saved_tensors
        shift = np.ones_like(inp.tolist()) * ctx.shift
        grads = []
        for val in inp.tolist():
            right = ctx.circuit.run([val + shift[0]])
            left = ctx.circuit.run([val - shift[0]])
            grads.append(right - left)
        grads = torch.tensor([grads]).float()
        return grads * grad_out.float(), None, None

class HybridAutoEncoderQCNet(nn.Module):
    """Hybrid quantum‑classical classifier mirroring the classical variant but with a quantum head."""
    def __init__(self, latent_dim: int = 32, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Autoencoder encoder
        self.autoencoder = build_autoencoder(input_dim=15, latent_dim=latent_dim)
        self.fc1 = nn.Linear(latent_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit = QuantumCircuitWrapper(1, backend, shots=100)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        z = self.autoencoder.encode(x)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = QuantumHybridLayer.apply(x, self.quantum_circuit, self.shift)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["HybridAutoEncoderQCNet"]
