import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 use_quantum_latent: bool = False):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_quantum_latent = use_quantum_latent

class ClassicalSelfAttention:
    """Simple self‑attention implementation matching the reference."""
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim
    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridAutoencoder(nn.Module):
    """Hybrid classical autoencoder optionally using a quantum latent vector."""
    def __init__(self, config: HybridAutoencoderConfig):
        super().__init__()
        self.config = config
        self.use_quantum_latent = config.use_quantum_latent
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        if self.use_quantum_latent:
            # output rotation and entangle params for the quantum circuit
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim * 2))
        else:
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Self‑attention block
        self.attention = ClassicalSelfAttention(embed_dim=4)
        # Decoder
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

    def _quantum_latent_sample(self, rotation: torch.Tensor, entangle: torch.Tensor):
        """Deterministic classical surrogate for a quantum latent sample."""
        return torch.sin(rotation) + torch.cos(entangle)

    def encode(self, inputs: torch.Tensor):
        x = self.encoder(inputs)
        if self.use_quantum_latent:
            rotation, entangle = torch.split(x, self.config.latent_dim, dim=-1)
            latent = self._quantum_latent_sample(rotation, entangle)
        else:
            latent = x
        # Apply self‑attention
        latent_np = latent.detach().cpu().numpy()
        attn_out = self.attention.run(np.zeros((1,)), np.zeros((1,)), latent_np)
        return torch.as_tensor(attn_out, dtype=latent.dtype, device=latent.device)

    def decode(self, latents: torch.Tensor):
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor):
        return self.decode(self.encode(inputs))

def HybridAutoencoderFactory(input_dim: int,
                            latent_dim: int = 32,
                            hidden_dims: tuple[int, int] = (128, 64),
                            dropout: float = 0.1,
                            use_quantum_latent: bool = False):
    cfg = HybridAutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout, use_quantum_latent)
    return HybridAutoencoder(cfg)

def train_hybrid_autoencoder(model: HybridAutoencoder,
                             data: torch.Tensor,
                             *,
                             epochs: int = 100,
                             batch_size: int = 64,
                             lr: float = 1e-3,
                             weight_decay: float = 0.0,
                             device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []
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

def _as_tensor(data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
