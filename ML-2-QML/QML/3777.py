import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

__all__ = ["UnifiedAutoencoder", "UnifiedAutoencoderConfig", "train_unified_autoencoder"]

class UnifiedAutoencoderConfig:
    """Configuration for the UnifiedAutoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1, weight_decay: float = 0.0,
                 quantum_shots: int = 512, quantum_shift: float = np.pi / 2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.quantum_shots = quantum_shots
        self.quantum_shift = quantum_shift

def identity_interpret(x):
    """Return raw expectation value."""
    return x

class QuantumLatentLayer(nn.Module):
    """Maps classical latent vector to quantum‑encoded features via a variational circuit."""
    def __init__(self, latent_dim: int, shots: int = 512, shift: float = np.pi / 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.shots = shots
        self.circuit = QuantumCircuit(latent_dim)
        # Input parameters (classical latent)
        self.input_params = []
        for i in range(latent_dim):
            theta = Parameter(f'theta_{i}')
            self.circuit.ry(theta, i)
            self.input_params.append(theta)
        # Variational layer
        self.circuit.compose(RealAmplitudes(latent_dim, reps=1).to_instruction(),
                             range(latent_dim), inplace=True)
        self.weight_params = list(self.circuit.parameters)[latent_dim:]
        self.circuit.measure_all()
        self.backend = Aer.get_backend('aer_simulator')
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=identity_interpret,
            output_shape=latent_dim,
            sampler=self.sampler,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        lat_np = latent.detach().cpu().numpy()
        q_lat = self.qnn(lat_np)
        return torch.tensor(q_lat, dtype=latent.dtype, device=latent.device)

class ClassicalEncoder(nn.Module):
    def __init__(self, config: UnifiedAutoencoderConfig):
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClassicalDecoder(nn.Module):
    def __init__(self, config: UnifiedAutoencoderConfig):
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class UnifiedAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder."""
    def __init__(self, config: UnifiedAutoencoderConfig, use_quantum: bool = True):
        super().__init__()
        self.encoder = ClassicalEncoder(config)
        self.decoder = ClassicalDecoder(config)
        self.use_quantum = use_quantum
        if use_quantum:
            self.quantum_latent = QuantumLatentLayer(
                config.latent_dim,
                shots=config.quantum_shots,
                shift=config.quantum_shift,
            )
        else:
            self.quantum_latent = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        lat = self.encoder(x)
        if self.use_quantum:
            return self.quantum_latent(lat)
        return lat

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def train_unified_autoencoder(
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
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
