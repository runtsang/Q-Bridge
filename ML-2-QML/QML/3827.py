"""Hybrid classical‑quantum binary classifier – quantum implementation.

The network mirrors the classical version but replaces the dense hybrid head
with a parameterised variational circuit evaluated on a quantum backend.
The autoencoder still operates classically to reduce the concatenated
feature vector before feeding the latent representation into the quantum
circuit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import assemble, transpile, BasicAer
from qiskit.circuit import Parameter, QuantumCircuit as QC
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler

# --------------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------------- #

class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = QC(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
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
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# Hybrid head – quantum expectation
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp = ctx.circuit.run(inputs.cpu().numpy())
        out = torch.tensor(exp, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for val, s in zip(inputs.cpu().numpy(), shift):
            grads.append(ctx.circuit.run([val + s])[0] - ctx.circuit.run([val - s])[0])
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x) if x.shape!= torch.Size([1, 1]) else x[0]
        return HybridFunction.apply(x, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# Autoencoder utilities – identical to the classical version
# --------------------------------------------------------------------------- #

def _as_tensor(data):
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(),
                           nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(),
                           nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Full network – convolution → autoencoder → quantum hybrid head
# --------------------------------------------------------------------------- #

class HybridAutoencoderQCNet(nn.Module):
    """CNN backbone, autoencoder, and quantum hybrid head."""
    def __init__(self, latent_dim: int = 4, autoencoder_hidden: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Autoencoder for dimensionality reduction
        self.autoencoder = Autoencoder(
            input_dim=55815,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout=0.1,
        )

        # Quantum hybrid head
        backend = BasicAer.get_backend("qasm_simulator")
        self.hybrid = Hybrid(latent_dim, backend, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Conv backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Autoencoder feature extraction
        latent = self.autoencoder.encode(x)

        # Apply quantum hybrid head
        logits = self.fc1(latent)
        logits = F.relu(logits)
        logits = self.fc2(logits)
        logits = F.relu(logits)
        logits = self.fc3(logits).squeeze(-1)  # shape (batch,)
        prob = self.hybrid(logits)

        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = [
    "Autoencoder", "AutoencoderNet", "AutoencoderConfig",
    "HybridFunction", "Hybrid", "HybridAutoencoderQCNet",
]
