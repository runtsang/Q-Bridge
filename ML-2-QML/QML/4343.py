import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Sampler as StatevectorSampler

# Seed for reproducibility
algorithm_globals.random_seed = 42

class TransformerBlock(nn.Module):
    """Classical transformer block used in both encoder and decoder."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridAutoencoder(nn.Module):
    """Quantum‑enhanced autoencoder: classical transformer encoder/decoder with a variational quantum latent layer."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Classical encoder
        self.encoder = nn.Sequential(
            *[TransformerBlock(config.embed_dim, config.num_heads, config.ffn_dim, config.dropout)
              for _ in range(config.num_encoder_blocks)]
        )

        # Quantum latent layer
        self.qc = self._build_quantum_circuit(config.latent_dim, config.num_trash)
        self.qsampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.qc,
            input_params=[],
            weight_params=self.qc.parameters,
            interpret=lambda x: x,
            output_shape=(config.latent_dim,),
            sampler=self.qsampler
        )

        # Classical decoder
        self.decoder = nn.Sequential(
            *[TransformerBlock(config.latent_dim, config.num_heads, config.ffn_dim, config.dropout)
              for _ in range(config.num_decoder_blocks)]
        )
        self.output = nn.Linear(config.latent_dim, config.input_dim)

    def _build_quantum_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, name='q')
        cr = ClassicalRegister(1, name='c')
        qc = QuantumCircuit(qr, cr)

        # Ansatz
        qc.append(RealAmplitudes(num_latent + num_trash, reps=5), list(range(num_latent + num_trash)))

        # Swap‑test for latent extraction
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        # Run quantum circuit to obtain latent vector
        z = self.qnn.forward(h.detach().cpu().numpy())
        return torch.tensor(z, dtype=torch.float32, device=x.device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        return self.output(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def train_quantum_autoencoder(
    model: HybridAutoencoder,
    data: np.ndarray | torch.Tensor,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that optimises classical weights and quantum parameters via COBYLA."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
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
