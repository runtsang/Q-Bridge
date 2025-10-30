from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit.utils import QuantumInstance

class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor inspired by EstimatorQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

class HybridAutoencoder(nn.Module):
    """
    Classical‑quantum autoencoder.
    The encoder is a variational RealAmplitudes circuit producing a
    probability‑amplitude vector via a StatevectorSampler.
    The decoder is a configurable MLP or an EstimatorNN.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        hidden_dims: tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        use_estimator: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_estimator = use_estimator

        # Quantum encoder
        self.qc = QuantumCircuit(latent_dim)
        self.qc.append(RealAmplitudes(latent_dim, reps=5), range(latent_dim))
        self.qc.save_statevector()
        self.sampler = StatevectorSampler(
            QuantumInstance(AerSimulator(method="statevector"))
        )

        # Classical decoder
        if use_estimator:
            self.decoder = EstimatorNN()
        else:
            layers = []
            in_dim = 2 ** latent_dim  # amplitude vector length
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, input_dim))
            self.decoder = nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode the input using the quantum circuit.
        The input is ignored because the encoder is purely parameterized;
        this mirrors the quantum autoencoder where the latent vector is
        derived from the circuit state.
        """
        # For each batch element, sample the statevector
        batch_size = inputs.shape[0]
        # Run the sampler once per batch; statevector same for all
        result = self.sampler.run(self.qc, shots=1)
        statevec = result.get_statevector()
        # Convert to probability amplitudes (real part)
        probs = np.abs(statevec) ** 2
        # Convert to torch tensor
        latent = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        # Expand to batch size
        latent = latent.unsqueeze(0).repeat(batch_size, 1)
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Standard reconstruction training loop.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["HybridAutoencoder", "EstimatorNN", "train_hybrid_autoencoder"]
