import numpy as np
import torch
from torch import nn
import qiskit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, PauliZ
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit import algorithm_globals
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

# Ensure reproducibility
algorithm_globals.random_seed = 42

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)
        ])
        self.pos_emb = PositionalEncoder(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        return x

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    embed_dim: int = 64
    num_heads: int = 4
    num_blocks: int = 2
    ffn_dim: int = 128
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    num_qubits: int = 4  # number of qubits used for quantum latent

class QuantumLatent(nn.Module):
    """Variational circuit producing a latent vector via Pauli‑Z expectations."""

    def __init__(self, num_qubits: int, latent_dim: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.circuit = self._build_circuit(num_qubits)
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=latent_dim,
            sampler=self.sampler,
        )

    def _build_circuit(self, num_qubits: int) -> qiskit.QuantumCircuit:
        qc = RealAmplitudes(num_qubits, reps=3)
        return qc

    def _interpret(self, state: np.ndarray) -> np.ndarray:
        sv = Statevector(state)
        expectations = []
        for i in range(self.num_qubits):
            exp = sv.expectation_value(PauliZ, qubits=[i])
            expectations.append(exp.real)
        # Pad or truncate to match latent_dim
        if len(expectations) < self.latent_dim:
            expectations += [0.0] * (self.latent_dim - len(expectations))
        return np.array(expectations)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, embed_dim)
        params = inputs[:, :self.num_qubits]
        params_np = params.detach().cpu().numpy()
        latent_np = self.qnn(np.array(params_np))
        return torch.tensor(latent_np, dtype=inputs.dtype, device=inputs.device)

class AutoencoderGen268Quantum(nn.Module):
    """Quantum‑augmented transformer autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
        self.quantum_latent = QuantumLatent(num_qubits=config.num_qubits, latent_dim=config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, config.embed_dim)

        decoder_layers = []
        in_dim = config.latent_dim
        for h in config.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return pooled

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = x.unsqueeze(1).repeat(1, 1, 1)
        return self.decoder(x).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        qz = self.quantum_latent(z)
        return self.decode(qz)

def train_autoencoder_quantum(
    model: AutoencoderGen268Quantum,
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["AutoencoderGen268Quantum", "train_autoencoder_quantum"]
