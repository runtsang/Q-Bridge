import torch
from torch import nn
import pennylane as qml
from typing import Iterable, Tuple

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input data into a float32 torch.Tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderGen265(nn.Module):
    """Hybrid quantum‑classical autoencoder.  
    The encoder is a variational quantum circuit that maps the input
    vector into a latent vector of Z‑expectation values.  
    The decoder is a small classical MLP that reconstructs the input.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 8,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        device: str | torch.device = 'cpu',
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device if isinstance(device, str) else device)

        # Quantum part
        self.n_qubits = latent_dim
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.n_layers = 2
        self.q_params = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 3))

        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], input_dim),
        )

    def encoder_qnode(self, x, params):
        qml.AngleEmbedding(x, wires=range(self.n_qubits))
        qml.templates.RealAmplitudes(params, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        qnode = qml.QNode(self.encoder_qnode, self.dev, interface="torch")
        latent = qnode(x, self.q_params)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def AutoencoderGen265_factory(
    input_dim: int,
    *,
    latent_dim: int = 8,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    device: str | torch.device = 'cpu',
) -> AutoencoderGen265:
    return AutoencoderGen265(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
    )

def train_autoencoder_qml(
    model: AutoencoderGen265,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid model.  
    The optimizer updates both the quantum parameters (self.q_params)
    and the classical decoder weights.  
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        list(model.parameters()), lr=lr, weight_decay=weight_decay
    )
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

__all__ = [
    "AutoencoderGen265",
    "AutoencoderGen265_factory",
    "train_autoencoder_qml",
]
