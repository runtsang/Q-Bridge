"""Python module implementing a hybrid auto‑encoder that uses a quantum circuit to refine the latent code.

The network consists of a classical encoder, a quantum refiner implemented
as a :class:`SamplerQNN`, and a classical decoder.  The forward pass
encodes the input, refines the latent vector with the quantum circuit,
and decodes the result.  The training loop optimises all three
components jointly using a reconstruction MSE loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def _build_quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a circuit that implements a swap‑test style auto‑encoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # encode the latent vector into the quantum state
    circuit.compose(
        RealAmplitudes(num_latent + num_trash, reps=1),
        range(0, num_latent + num_trash),
        inplace=True,
    )
    circuit.barrier()

    # swap‑test
    circuit.h(0)  # auxiliary qubit
    for i in range(num_trash):
        circuit.cswap(0, num_latent + i, num_trash + i)
    circuit.h(0)
    circuit.measure(0, cr[0])

    return circuit

def _interpret(*args: float | int) -> float:
    """Return the probability of measuring 0 from the auxiliary qubit."""
    return float(args[0])

class UnifiedAutoEncoder:
    """Hybrid auto‑encoder that uses a quantum circuit to refine the latent code.

    The network consists of a classical encoder, a quantum refiner implemented
    as a :class:`SamplerQNN`, and a classical decoder.  The forward pass
    encodes the input, refines the latent vector with the quantum circuit,
    and decodes the result.  The training loop optimises all three
    components jointly using a reconstruction MSE loss.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 num_trash: int = 2,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")

        # Classical encoder
        self.encoder = nn.Linear(input_dim, latent_dim).to(self.device)

        # Quantum refiner
        self.circuit = _build_quantum_autoencoder_circuit(latent_dim, num_trash)
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters(),
            interpret=_interpret,
            output_shape=1,
            sampler=self.sampler,
        ).to(self.device)

        # Classical decoder
        self.decoder = nn.Linear(latent_dim, input_dim).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x.to(self.device))
        probs = self.qnn(z)  # shape (batch, 1)
        refined = z * probs  # broadcast to (batch, latent_dim)
        return self.decoder(refined)

    def train(self,
              data: torch.Tensor,
              *,
              epochs: int = 100,
              batch_size: int = 64,
              lr: float = 1e-3,
              weight_decay: float = 0.0) -> list[float]:
        self.encoder.train()
        self.decoder.train()
        self.qnn.train()
        dataset = TensorDataset(data.to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.qnn.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.forward(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

__all__ = ["UnifiedAutoEncoder"]
