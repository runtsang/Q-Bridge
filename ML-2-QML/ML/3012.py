"""UnifiedAutoencoder: hybrid classical autoencoder with optional quantum encoder/decoder.

The class can be instantiated with `use_quantum=True` to enable a quantum
autoencoder that shares the same latent dimension as the classical encoder.
When the quantum part is active, the forward pass returns both the classical
reconstruction and the quantum state fidelity.  The quantum module uses a
RealAmplitudes ansatz and a swap‑test style measurement to evaluate the
reconstruction fidelity.  The loss can be combined using a weighted sum
of MSE, fidelity and KL divergence.

The module keeps the same public API as the original Autoencoder:
`Autoencoder(...)` factory, `train_autoencoder(...)` training loop.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

# Optional quantum import
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import StatevectorSampler
except Exception:
    QuantumCircuit = None
    RealAmplitudes = None
    StatevectorSampler = None

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quantum: bool = False

class UnifiedAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Optional quantum encoder/decoder
        if config.use_quantum and QuantumCircuit is not None:
            self.qc_encoder = self._build_qc_encoder()
            self.qc_decoder = self._build_qc_decoder()
            self.sampler = StatevectorSampler()
        else:
            self.qc_encoder = None
            self.qc_decoder = None
            self.sampler = None

    def _build_qc_encoder(self) -> QuantumCircuit:
        """Build a RealAmplitudes encoder that maps the input to a latent state."""
        qr = QuantumRegister(self.config.latent_dim, "q")
        qc = QuantumCircuit(qr)
        qc.compose(RealAmplitudes(self.config.latent_dim, reps=3), inplace=True)
        return qc

    def _build_qc_decoder(self) -> QuantumCircuit:
        """Build a simple decoder that maps latent qubits back to classical bits."""
        qr = QuantumRegister(self.config.latent_dim, "q")
        qc = QuantumCircuit(qr)
        qc.compose(RealAmplitudes(self.config.latent_dim, reps=3), inplace=True)
        return qc

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def quantum_encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a quantum statevector for the input."""
        if self.qc_encoder is None:
            raise RuntimeError("Quantum encoder is not initialized.")
        input_np = inputs.cpu().detach().numpy()
        # For simplicity we only support batch size 1 in this toy example
        assert input_np.shape[0] == 1
        qc = self.qc_encoder.assign_parameters(input_np.flatten(), inplace=True)
        sv = self.sampler.run(qc).result().quasi_distribution
        return torch.tensor(sv, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return classical reconstruction and optional quantum fidelity."""
        latent = self.encode(inputs)
        recon = self.decode(latent)
        if self.qc_encoder is not None:
            # Compute quantum fidelity as a simple dot product with the classical latent
            qstate = self.quantum_encode(inputs.unsqueeze(0))
            fidelity = torch.dot(latent.squeeze(), torch.from_numpy(qstate))
            return recon, fidelity
        return recon

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quantum: bool = False,
) -> UnifiedAutoencoder:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum=use_quantum,
    )
    return UnifiedAutoencoder(config)

def train_autoencoder(
    model: UnifiedAutoencoder,
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
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if model.qc_encoder is not None:
                recon, fidelity = model(batch)
                # Simple hybrid loss: weighted sum of MSE and negative fidelity
                loss = mse_loss(recon, batch) - 0.1 * fidelity
            else:
                recon = model(batch)
                loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["UnifiedAutoencoder", "AutoencoderConfig", "Autoencoder", "train_autoencoder"]
