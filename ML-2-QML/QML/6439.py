"""Hybrid autoencoder with a quantum variational encoder and a classical decoder.

The encoder is a variational circuit inspired by Autoencoder.py's
`auto_encoder_circuit`, augmented with a quantum convolution filter
taken from Conv.py.  The decoder is a simple linear layer, allowing
side‑by‑side benchmarking against the classical counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import qiskit
from qiskit import QuantumCircuit, Parameter, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.aer import Aer


# Quantum convolution filter from Conv.py
class QuanvCircuit:
    """Quantum filter that maps a 2×2 patch to a single probability value."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, depth=2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: Iterable[float]) -> float:
        """Return average probability of measuring |1> across qubits."""
        param_binds = []
        for val in data:
            bind = {self.theta[i]: (torch.pi if val > self.threshold else 0) for i in range(self.n_qubits)}
            param_binds.append(bind)

        job = execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)
        counts = sum(int(bit) * freq for key, freq in result.items() for bit in key)
        return counts / (self.shots * self.n_qubits)


def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Variational circuit used to encode a latent vector."""
    qr = QuantumCircuit(num_latent + 2 * num_trash + 1)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qr.append(ansatz, range(num_latent + num_trash))
    qr.barrier()
    aux = num_latent + 2 * num_trash
    qr.h(aux)
    for i in range(num_trash):
        qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qr.h(aux)
    qr.measure_all()
    return qr


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class HybridAutoencoder(nn.Module):
    """Quantum‑classical hybrid autoencoder."""

    def __init__(
        self,
        config: AutoencoderConfig,
        *,
        backend=None,
        shots: int = 1024,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Quantum encoder circuit with a placeholder parameter
        self.latent_dim = config.latent_dim
        self.base_circuit = _auto_encoder_circuit(self.latent_dim, num_trash=2)
        self.theta = Parameter("theta")
        # Insert a parameterised rotation on qubit 0
        self.base_circuit.cx(0, 1)  # dummy gate to host the parameter
        self.base_circuit.rx(self.theta, 0)

        # Quantum convolution filter
        self.conv_filter = QuanvCircuit(
            kernel_size=2,
            backend=self.backend,
            shots=self.shots,
            threshold=0.5,
        )

        # Sampler for the encoder
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.base_circuit,
            input_params=[self.theta],
            weight_params=[],
            interpret=lambda x: x,
            output_shape=(self.latent_dim,),
            sampler=self.sampler,
        )

        # Classical decoder
        self.decoder = nn.Linear(self.latent_dim, config.input_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs using the quantum circuit."""
        latents = []
        for sample in inputs:
            patch = sample.view(-1).numpy()
            feature = self.conv_filter.run(patch)
            bound_circuit = self.base_circuit.bind_parameters({self.theta: feature * torch.pi})
            result = self.sampler.run(bound_circuit).result()
            counts = result.get_counts(bound_circuit)
            exp_val = sum(
                (int(bit) * freq) for key, freq in counts.items() for bit in key
            ) / (self.shots * self.base_circuit.num_qubits)
            latents.append(torch.full((self.latent_dim,), exp_val, device=self.device))
        return torch.stack(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    backend=None,
    shots: int = 1024,
) -> HybridAutoencoder:
    """Factory mirroring the classical helper."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(cfg, backend=backend, shots=shots)


def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 10,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder using a classical optimizer."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        recon = model(data.to(device))
        loss = loss_fn(recon, data.to(device))
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "HybridAutoencoder",
    "train_autoencoder",
]
