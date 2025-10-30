"""Hybrid autoencoder + quantum classifier for binary image classification.

This module builds on a fully‑connected autoencoder to compress raw inputs
before passing the latent representation to a variational quantum layer.
The quantum head is implemented with a two‑qubit circuit and a
differentiable expectation value, enabling end‑to‑end training with
PyTorch autograd.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer
from qiskit import assemble, transpile


# --------------------------------------------------------------------------- #
# Classical autoencoder (from reference pair 2)
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, hidden),
                               nn.ReLU(),
                               nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, hidden),
                               nn.ReLU(),
                               nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Quantum expectation head (from reference pair 1)
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """Two‑qubit parametrised circuit executed on Aer."""
    def __init__(self, backend=Aer.get_backend('aer_simulator'), shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = None
        self.theta = None

    def build(self, n_params: int):
        """Build a simple parameterised circuit with H and RY layers."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.barrier()
        self.theta = [qc.circuit.Parameter(f'theta_{i}') for i in range(n_params)]
        for i, p in enumerate(self.theta):
            qc.ry(p, i % 2)
        qc.measure_all()
        self.circuit = qc

    def run(self, params: np.ndarray) -> np.ndarray:
        if params.ndim == 1:
            param_list = [params]
        else:
            param_list = params.tolist()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_list)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts0 = count_dict.get('0', 0)
            counts1 = count_dict.get('1', 0)
            return (counts0 - counts1) / self.shots
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        else:
            return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        params = inputs.detach().cpu().numpy()
        exp = circuit.run(params)
        out = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for i, val in enumerate(inputs.detach().cpu().numpy()):
            right = ctx.circuit.run([val + shift[i]])
            left = ctx.circuit.run([val - shift[i]])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None


class HybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_params: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper()
        self.circuit.build(n_params)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
# Hybrid autoencoder + quantum classifier
# --------------------------------------------------------------------------- #
class HybridAutoencoderClassifier(nn.Module):
    """End‑to‑end model: autoencoder for feature reduction, quantum head for classification."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 n_params: int | None = None):
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout)
        self.n_params = n_params if n_params is not None else latent_dim
        self.quantum_head = HybridLayer(n_params=self.n_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        latent = self.autoencoder.encode(x)
        logits = self.quantum_head(latent)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["AutoencoderNet", "HybridLayer", "HybridAutoencoderClassifier"]
