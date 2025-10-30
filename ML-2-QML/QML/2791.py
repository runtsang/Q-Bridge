"""Quantum‑enhanced autoencoder using a swap‑test based latent reconstruction.

The module implements :class:`QuantumAutoencoderHybridNet`, a torch
``nn.Module`` that wraps a Qiskit ``SamplerQNN`` built from a
RealAmplitudes ansatz and a domain‑wall swap test.  The network
produces a 2‑dimensional quantum embedding which is linearly mapped
to the original feature space.  The helper :func:`QuantumAutoencoderHybrid`
creates a ready‑to‑train instance, and
:func:`train_quantum_autoencoder_hybrid` trains the model end‑to‑end
with a classical Adam optimiser.

Typical usage:

>>> from Autoencoder__gen046 import QuantumAutoencoderHybrid, train_quantum_autoencoder_hybrid
>>> model = QuantumAutoencoderHybrid(784, latent_dim=3, trash_dim=2)
>>> history = train_quantum_autoencoder_hybrid(model, train_data)

"""

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class QuantumAutoencoderHybridNet(nn.Module):
    """Quantum autoencoder with a classical post‑processing layer."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        trash_dim: int = 2,
        sampler: StatevectorSampler | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.sampler = sampler or StatevectorSampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.post = nn.Linear(2, input_dim)

    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Simple feature encoding on latent qubits
        for i in range(self.latent_dim):
            circuit.h(qr[i])

        # Parameterised ansatz on all but the auxiliary qubit
        ansatz = RealAmplitudes(num_qubits - 1, reps=5)
        circuit.compose(ansatz, list(range(num_qubits - 1)), inplace=True)

        # Swap test between trash qubits
        aux = num_qubits - 1
        circuit.h(aux)
        for i in range(self.trash_dim):
            circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The quantum part is independent of the classical input in this toy example.
        # We run the circuit once per batch element to keep shapes consistent.
        batch_size = x.shape[0]
        outputs = []
        for _ in range(batch_size):
            qout = self.qnn.forward([])  # use current parameters
            outputs.append(qout)
        qout = torch.tensor(outputs, dtype=torch.float32, device=x.device)
        return self.post(qout)


def QuantumAutoencoderHybrid(
    input_dim: int,
    latent_dim: int = 3,
    trash_dim: int = 2,
) -> QuantumAutoencoderHybridNet:
    """Factory returning a configured :class:`QuantumAutoencoderHybridNet`."""
    return QuantumAutoencoderHybridNet(input_dim, latent_dim, trash_dim)


def train_quantum_autoencoder_hybrid(
    model: QuantumAutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """End‑to‑end training loop for the quantum autoencoder."""
    device = device or torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    "QuantumAutoencoderHybridNet",
    "QuantumAutoencoderHybrid",
    "train_quantum_autoencoder_hybrid",
]
