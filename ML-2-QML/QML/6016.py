from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler


def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class QuantumHybridAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        n_qubits: int | None = None,
        reps: int = 3,
        sampler: StatevectorSampler | None = None,
    ) -> None:
        super().__init__()
        if n_qubits is None:
            n_qubits = latent_dim
        self.n_qubits = n_qubits

        # Feature map to embed classical data into a quantum state
        self.feature_map = RawFeatureVector(input_dim)

        # Variational encoder to produce a latent state
        self.encoder = RealAmplitudes(n_qubits, reps=reps)

        # Composite circuit: feature map followed by encoder
        circuit = QuantumCircuit(input_dim + n_qubits)
        circuit.compose(self.feature_map, range(input_dim), inplace=True)
        circuit.compose(self.encoder, range(input_dim, input_dim + n_qubits), inplace=True)

        # QNN that samples expectation values of Z on each qubit
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.encoder.parameters,
            interpret=lambda x: x,
            output_shape=n_qubits,
            sampler=sampler or StatevectorSampler(),
        )

        # Classical decoder mapping latent vector back to input space
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        latent = []
        for i in range(batch_size):
            input_vals = inputs[i].detach().cpu().numpy()
            # For demonstration the encoder weights are fixed at zero
            weight_vals = np.zeros(len(self.encoder.parameters))
            out = self.qnn(weight_vals, input_vals)
            latent.append(out)
        latent = torch.tensor(latent, dtype=torch.float32, device=inputs.device)
        return self.decoder(latent)


def QuantumHybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    n_qubits: int | None = None,
    reps: int = 3,
    sampler: StatevectorSampler | None = None,
) -> QuantumHybridAutoencoder:
    return QuantumHybridAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_qubits=n_qubits,
        reps=reps,
        sampler=sampler,
    )


def train_quantum_autoencoder(
    model: QuantumHybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["QuantumHybridAutoencoder", "QuantumHybridAutoencoderFactory", "train_quantum_autoencoder"]
