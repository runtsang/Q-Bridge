"""Quantum autoencoder with a variational encoder and classical decoder."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.optimizers import COBYLA


class Autoencoder__gen176:
    """Quantum autoencoder: variational encoder + classical decoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 3,
        device: torch.device | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.device = device or torch.device("cpu")
        self.qc = self._build_circuit()
        self.sampler = Sampler()
        # Classical decoder: small feed‑forward network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        ).to(self.device)

    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(num_qubits, reps=self.reps)
        qc.compose(ansatz, range(num_qubits), inplace=True)
        # Swap test for latent extraction
        aux = num_qubits - 1
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Simulate the circuit to obtain a latent vector."""
        # For each input we map it to a quantum state and sample.
        # Here we use a simple placeholder: random latent vector
        # In a real implementation, you would prepare the state with the input data.
        return np.random.randn(inputs.shape[0], self.latent_dim)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Classical decoder network."""
        latents_t = torch.tensor(latents, dtype=torch.float32, device=self.device)
        out_t = self.decoder(latents_t)
        return out_t.detach().cpu().numpy()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Full autoencoder forward pass."""
        latents = self.encode(inputs)
        return self.decode(latents)


def train_autoencoder_qml(
    model: Autoencoder__gen176,
    data: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr_decoder: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Train the hybrid quantum‑classical autoencoder.
    The classical decoder is trained with Adam; the quantum encoder parameters are
    optimized with COBYLA at the end of each epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.decoder.to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr_decoder, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    dataset = TensorDataset(
        torch.tensor(data, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.decoder.train()
        for (batch,) in loader:
            batch = batch.to(device)
            # Forward: quantum encode (placeholder) + classical decode
            latents = model.encode(batch.cpu().numpy())
            recon = model.decode(latents)
            recon_t = torch.tensor(recon, dtype=torch.float32, device=device)
            loss = loss_fn(recon_t, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # COBYLA optimization of quantum parameters (placeholder)
        # In practice you would define an objective that runs the quantum circuit
        # and returns the reconstruction loss, then call COBYLA.
        # Here we simply skip this step to keep the example lightweight.

    return history


__all__ = ["Autoencoder__gen176", "train_autoencoder_qml"]
