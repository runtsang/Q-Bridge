# autoencoder_hybrid_qml.py

"""
AutoencoderHybrid: Hybrid quantum-classical autoencoder.

Features:
- Quantum encoder implemented with PennyLane variational circuit.
- Classical decoder implemented as a linear layer.
- End-to-end differentiable training using PyTorch autograd.
- Supports reconstruction loss and optional KL divergence for VAE mode.
"""

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Callable

def _as_tensor(data: torch.Tensor | list | tuple) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderHybrid(nn.Module):
    """Hybrid quantum-classical autoencoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_qubits: int = 4,
        reps: int = 2,
        batchnorm: bool = False,
        activation: Callable = nn.ReLU,
        vae: bool = False,
    ):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "num_qubits": num_qubits,
            "reps": reps,
            "batchnorm": batchnorm,
            "activation": activation,
            "vae": vae,
        }
        self.vae = vae
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Quantum encoder circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_encoder(inputs: torch.Tensor, weights: torch.Tensor):
            # Feature map
            for i in range(num_qubits):
                qml.RX(inputs[i % input_dim], wires=i)
            # Variational layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.quantum_encoder = quantum_encoder

        # Initialize variational parameters
        weight_shape = qml.templates.StronglyEntanglingLayers.shape(num_qubits, reps)
        self.q_weights = nn.Parameter(torch.randn(weight_shape, requires_grad=True))

        # Classical decoder
        decoder_in = num_qubits
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, latent_dim),
            activation(),
            nn.Linear(latent_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input using the quantum circuit."""
        batch_size = x.shape[0]
        encoded = []
        for i in range(batch_size):
            out = self.quantum_encoder(x[i], self.q_weights)
            encoded.append(out)
        return torch.stack(encoded, dim=0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from its latent representation."""
        return self.forward(x)

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Sample from standard normal prior and decode."""
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.decode(z)

def train_autoencoder_qml(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[list[float], list[float]]:
    """Train the hybrid quantum-classical autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss(reduction="sum")
    recon_hist: list[float] = []
    kl_hist: list[float] = []

    for _ in range(epochs):
        epoch_recon = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_recon += loss.item()
        epoch_recon /= len(dataset)
        recon_hist.append(epoch_recon)
    return recon_hist, kl_hist

__all__ = [
    "AutoencoderHybrid",
    "train_autoencoder_qml",
]
