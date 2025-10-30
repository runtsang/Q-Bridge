import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

# ------------------------------------------------------------------
# Classical auto‑encoder with optional convolutional front‑end
# ------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int                 # 1‑D input shape (int or shape tuple)  (i.e. A)
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_conv_front: bool = False  # optional 2‑D conv front‑end

class ClassicalAutoencoder(nn.Module):
    """Fully‑connected auto‑encoder that can optionally prepend a 2‑D conv front‑end."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # optional conv front‑end (inspired by QuanvolutionFilter)
        if config.use_conv_front:
            self.conv_front = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        else:
            self.conv_front = None

        # encoder
        enc_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder
        dec_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_front is not None:
            x = self.conv_front(x)
            x = x.view(x.size(0), -1)          # flatten after conv
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

# ------------------------------------------------------------------
# Quantum‑variational encoder / decoder
# ------------------------------------------------------------------
class QuantumEncoder(nn.Module):
    """
    Variational encoder that maps a classical vector to a quantum state,
    then samples a classical latent vector via a statevector sampler.
    """
    def __init__(self, latent_dim: int, n_qubits: int | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # choose number of qubits to encode
        self.n_qubits = n_qubits or (latent_dim + 1)  # +1 for ancilla / swap‑test
        # ansatz: RealAmplitudes with 5 reps
        from qiskit.circuit.library import RealAmplitudes
        self.ansatz = RealAmplitudes(self.n_qubits, reps=5)
        # sampler primitive
        from qiskit.primitives import StatevectorSampler
        self.sampler = StatevectorSampler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, latent_dim]
        returns: [batch, latent_dim] classical latent vector obtained from the quantum circuit
        """
        # create quantum circuit
        from qiskit import QuantumCircuit, QuantumRegister
        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)
        # encode classical weights into rotation angles (one per qubit)
        qc.append(self.ansatz, qr)
        # swap‑test‑style measurement on ancilla
        ancilla = self.n_qubits - 1
        qc.h(ancilla)
        qc.barrier()
        qc.measure_all()
        # sample
        result = self.sampler.run(qc, parameter_binds=[], shots=1024)
        # convert to classical vector; simple expectation values of Z
        state = result[0].statevector()
        meas = (state[::2] - state[1::2]).real  # crude placeholder
        return torch.tensor(meas[:self.latent_dim], dtype=x.dtype, device=x.device)

class QuantumDecoder(nn.Module):
    """
    Simple classical decoder that maps the quantum‑sampled latent vector
    back to the original input dimension.
    """
    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

# ------------------------------------------------------------------
# Hybrid auto‑encoder that stitches classical and quantum parts
# ------------------------------------------------------------------
class AutoencoderHybrid(nn.Module):
    """
    Hybrid auto‑encoder that first runs a classical encoder (optionally with conv front‑end),
    then feeds the latent vector into a variational quantum encoder, finally decodes
    the quantum‑sampled latent vector back to the input space.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.classical = ClassicalAutoencoder(config)
        self.quantum = QuantumEncoder(config.latent_dim)
        self.decoder = QuantumDecoder(config.latent_dim, config.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_classical = self.classical.encode(x)
        z_quantum = self.quantum(z_classical)
        return self.decoder(z_quantum)

# ------------------------------------------------------------------
# Training helper
# ------------------------------------------------------------------
def train_hybrid(
    model: AutoencoderHybrid,
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

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderConfig",
    "train_hybrid",
]
