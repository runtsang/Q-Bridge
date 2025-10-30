"""Hybrid classical‑quantum autoencoder with attention and quantum latent space.

The module defines :class:`AutoencoderHybridNet`, a PyTorch neural network that
encodes inputs through a multi‑head self‑attention layer, maps the resulting
classical latent vector to a quantum state using a :class:`SamplerQNN`,
and decodes back to the input space.  The architecture is inspired by the
separate Autoencoder, SelfAttention, and EstimatorQNN seeds, but unifies them
into a single end‑to‑end model.

Usage
-----
>>> from Autoencoder__gen175 import AutoencoderHybrid
>>> model = AutoencoderHybrid(input_dim=784, latent_dim=16)
>>> loss_hist = train_autoencoder_hybrid(model, data, epochs=50)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Quantum‑classical bridge
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler

# --------------------------------------------------------------------------- #
#  Classical attention helper
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    """Simple multi‑head self‑attention block.

    The implementation mirrors the classical SelfAttention seed but
    generalises to an arbitrary number of heads and allows the head
    dimension to be tuned independently.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of the attention block, same shape as input.
        """
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # each : (B, N, heads, head_dim)

        # Scaled dot‑product attention per head
        scores = torch.einsum("bnhd,bmhd->bnhm", q, k) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        out = torch.einsum("bnhm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        return out

# --------------------------------------------------------------------------- #
#  Quantum latent encoder
# --------------------------------------------------------------------------- #
def _build_quantum_latent_circuit(latent_dim: int, trash_dim: int, reps: int = 2) -> QuantumCircuit:
    """
    Construct a hybrid circuit that encodes a classical latent vector into a
    quantum state with an auxiliary swap‑test for fidelity estimation.

    Parameters
    ----------
    latent_dim : int
        Number of qubits representing the latent vector.
    trash_dim : int
        Additional qubits used as ancillary “trash” for entanglement.
    reps : int
        Number of repetitions of the RealAmplitudes ansatz.

    Returns
    -------
    QuantumCircuit
        The circuit ready for use with :class:`SamplerQNN`.
    """
    total = latent_dim + 2 * trash_dim + 1  # +1 for swap‑test ancilla
    qr = QuantumRegister(total, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the latent part with a parameterised ansatz
    ansatz = RealAmplitudes(latent_dim + trash_dim, reps=reps)
    qc.compose(ansatz, range(latent_dim + trash_dim), inplace=True)

    # Entangle latent and trash qubits with controlled‑X gates
    for i in range(trash_dim):
        qc.cx(latent_dim + i, latent_dim + trash_dim + i)

    # Swap‑test for fidelity estimation
    anc = latent_dim + 2 * trash_dim
    qc.h(anc)
    for i in range(trash_dim):
        qc.cswap(anc, latent_dim + i, latent_dim + trash_dim + i)
    qc.h(anc)
    qc.measure(anc, cr[0])

    return qc

# --------------------------------------------------------------------------- #
#  Hybrid autoencoder model
# --------------------------------------------------------------------------- #
class AutoencoderHybridNet(nn.Module):
    """Hybrid classical‑quantum autoencoder.

    Architecture
    ------------
    * Classical encoder : linear → ReLU → dropout → attention → linear → latent
    * Quantum latent : SamplerQNN with RealAmplitudes ansatz
    * Classical decoder : linear → ReLU → dropout → linear → reconstruction
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        attention_heads: int = 4,
        trash_dim: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout)]
        encoder_layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.append(nn.Linear(hidden_dims[1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Attention on latent representation
        self.attention = MultiHeadSelfAttention(embed_dim=latent_dim, num_heads=attention_heads)

        # Quantum latent
        qc = _build_quantum_latent_circuit(latent_dim=latent_dim, trash_dim=trash_dim)
        sampler = StatevectorSampler()
        self.quantum_latent = SamplerQNN(
            circuit=qc,
            input_params=[],            # no classical inputs
            weight_params=qc.parameters,
            interpret=lambda x: x,       # raw probability amplitudes
            output_shape=latent_dim,
            sampler=sampler,
        )

        # Decoder
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], input_dim),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Classical encoder followed by quantum latent."""
        latent = self.encoder(x)
        # Reshape for attention: (batch, seq_len=1, embed_dim)
        latent = latent.unsqueeze(1)
        latent = self.attention(latent).squeeze(1)

        # Quantum forward pass
        # Convert to numpy array of parameters for SamplerQNN
        params = latent.detach().cpu().numpy()
        q_latent = self.quantum_latent.forward(params)
        return torch.tensor(q_latent, dtype=torch.float32, device=x.device)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def AutoencoderHybrid(input_dim: int, **kwargs) -> AutoencoderHybridNet:
    """Convenience factory mirroring the classical Autoencoder factory."""
    return AutoencoderHybridNet(input_dim=input_dim, **kwargs)


def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> list[float]:
    """Training loop for the hybrid autoencoder.

    The loss is the mean‑squared reconstruction error.  The quantum part
    is treated as a black‑box differentiable layer via the sampler
    back‑propagation provided by Qiskit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


__all__ = ["AutoencoderHybridNet", "AutoencoderHybrid", "train_autoencoder_hybrid"]
