import torch
from torch import nn
import numpy as np
from typing import Iterable
from quantum_layer import QuantumHybridAutoencoder as QuantumHybridAutoencoderQuantum

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class QuantumHybridAutoencoder(QuantumHybridAutoencoderQuantum):
    """
    Hybrid autoencoder that combines a classical encoder‑decoder with a quantum
    fully‑connected layer in the latent space. The quantum layer is treated as a
    black‑box function; gradients flow only through the classical parts.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 quantum_qubits: int | None = None,
                 quantum_backend: str = "qasm_simulator",
                 quantum_shots: int = 1024):
        super().__init__(n_qubits=quantum_qubits or latent_dim,
                         backend=quantum_backend,
                         shots=quantum_shots)
        self.latent_dim = latent_dim
        self.quantum_qubits = quantum_qubits or latent_dim

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        latent = self.encoder(x)
        # Quantum processing
        latent_np = latent.detach().cpu().numpy()
        quantum_out_np = self.run(latent_np)
        quantum_out = torch.from_numpy(quantum_out_np).to(latent.device)
        # Classical decoding
        return self.decoder(quantum_out)

    def train_hybrid_autoencoder(self,
                                 data: torch.Tensor,
                                 *,
                                 epochs: int = 100,
                                 batch_size: int = 64,
                                 lr: float = 1e-3,
                                 weight_decay: float = 0.0,
                                 device: torch.device | None = None) -> list[float]:
        """
        Simple reconstruction training loop returning the loss history.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = torch.utils.data.TensorDataset(_as_tensor(data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history
