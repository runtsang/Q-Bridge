import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

# --------------------------------------------------------------------------- #
# Classical Autoencoder Backbone
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for the classical autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with optional dropout."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def Autoencoder(input_dim: int,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    """Factory mirroring the original Autoencoder API."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Transformer‑Style Latent Encoder
# --------------------------------------------------------------------------- #

class PositionalEncoder(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """A minimal transformer block using classical multi‑head attention."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerEncoder(nn.Module):
    """Encodes a sequence of latent vectors into a richer representation."""
    def __init__(self,
                 latent_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 ffn_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(latent_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch, seq_len, latent_dim)
        for layer in self.layers:
            z = layer(z)
        return z


# --------------------------------------------------------------------------- #
# Quantum‑Enhanced Latent Representation
# --------------------------------------------------------------------------- #

class QuantumLatentModule(nn.Module):
    """
    Variational circuit that maps a classical latent vector to a quantum state
    and samples a new latent representation.  The circuit is built from
    RealAmplitudes style blocks, mirroring the QML seed's swap‑test construction.
    """
    def __init__(self,
                 latent_dim: int,
                 num_trash: int = 2,
                 reps: int = 5,
                 q_device: Optional[object] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.q_device = q_device or self._default_device()

        # Build a RealAmplitudes ansatz with the required qubit count.
        from qiskit.circuit.library import RealAmplitudes
        self.ansatz = RealAmplitudes(latent_dim + num_trash, reps=reps)

    def _default_device(self):
        try:
            from qiskit import Aer
            return Aer.get_backend('statevector_simulator')
        except Exception:
            return None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim)  ->  (batch, latent_dim)
        """
        import numpy as np
        from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
        from qiskit.quantum_info import Statevector

        batch = z.size(0)
        # Convert to numpy and build a circuit per batch element
        outputs = []
        for i in range(batch):
            vec = z[i].detach().cpu().numpy()
            # Prepare initial state
            qc = QuantumCircuit(self.latent_dim + self.num_trash)
            qc.initialize(vec / np.linalg.norm(vec + 1e-8), self.latent_dim + self.num_trash - 1)
            qc.append(self.ansatz, range(0, self.latent_dim + self.num_trash))
            # quantum measurement
            sv = Statevector.from_instruction(qc)
            # Sample 2‑bit measurement for each trash qubit
            output = sv.data.real
            # 1‑D array; reshape to match latent_dim
            output = torch.tensor(output[:self.latent_dim], dtype=torch.float32)
            outputs.append(output)
        return torch.stack(outputs, dim=0)


# --------------------------------------------------------------------------- #
# Unified Hybrid Model
# --------------------------------------------------------------------------- #

class UnifiedAutoencoderTransformer(nn.Module):
    """
    Combines a classical autoencoder, a transformer‑style latent encoder,
    and a quantum‑enhanced latent module.  The forward pass first
    reconstructs the input, then passes the latent code through the
    transformer and quantum modules to produce a latent representation
    that can be used for downstream tasks or generation.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 transformer_layers: int = 2,
                 transformer_heads: int = 4,
                 transformer_ffn: int = 128,
                 num_trash: int = 2,
                 quantum_reps: int = 5,
                 q_device: Optional[object] = None):
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim, hidden_dims, dropout)
        self.transformer = TransformerEncoder(
            latent_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ffn_dim=transformer_ffn,
            dropout=dropout,
        )
        self.quantum = QuantumLatentModule(
            latent_dim,
            num_trash=num_trash,
            reps=quantum_reps,
            q_device=q_device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical autoencoding
        z = self.autoencoder.encode(x)
        # Treat each latent dimension as a token for transformer
        seq = z.unsqueeze(1)  # (batch, 1, latent_dim)
        z = self.transformer(seq).squeeze(1)
        # Quantum enhancement
        z = self.quantum(z)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum(self.transformer(self.autoencoder.encode(x).unsqueeze(1)).squeeze(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)

__all__ = [
    "UnifiedAutoencoderTransformer",
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "TransformerEncoder",
    "TransformerBlock",
    "PositionalEncoder",
    "QuantumLatentModule",
]
