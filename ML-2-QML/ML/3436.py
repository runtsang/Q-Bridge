"""
Hybrid Autoencoder combining classical MLPs with optional quantum encoder/decoder.
The module is fully importable and uses only optional Qiskit imports, so it works
even if Qiskit is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn

# Optional quantum imports – they are imported lazily to keep the module usable
# without a Qiskit installation
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_machine_learning.neural_networks import SamplerQNN
except Exception:  # pragma: no cover
    QuantumCircuit = None
    QuantumRegister = None
    ClassicalRegister = None
    Aer = None
    RealAmplitudes = None
    SamplerQNN = None


# --------------------------------------------------------------------------- #
# Classical Self‑Attention (mirrors the quantum interface)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    A lightweight self‑attention block that mimics the interface of the
    quantum SelfAttention helper.  It is fully differentiable and can be
    dropped in place of the quantum version when training the classical
    part of the hybrid model.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a self‑attention weighted sum of the inputs.

        Args:
            inputs: (batch, features)
            rotation_params: Tensor reshaped to (embed_dim, -1)
            entangle_params: Tensor reshaped to (embed_dim, -1)

        Returns:
            Tensor of shape (batch, features)
        """
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


# --------------------------------------------------------------------------- #
# Quantum Self‑Attention helper (used to build the quantum encoder)
# --------------------------------------------------------------------------- #
def _build_qiskit_self_attention(
    num_qubits: int, rotation_params: np.ndarray, entangle_params: np.ndarray
) -> QuantumCircuit:
    """
    Build a small quantum circuit that implements a self‑attention style
    block using RX, RY, RZ rotations followed by a chain of controlled‑RX gates.
    The circuit ends with a measurement of all qubits.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    for i in range(num_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    for i in range(num_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration values for the hybrid Autoencoder."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    mode: str = "hybrid"  # options: "classical", "quantum", "hybrid"


# --------------------------------------------------------------------------- #
# Core hybrid model
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """
    Hybrid autoencoder that can operate in classical, quantum or hybrid mode.
    In hybrid mode, the latent representation is the sum of a classical
    MLP latent vector and a quantum latent vector produced by a SamplerQNN.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        if config.mode in ("classical", "hybrid"):
            self.classical_encoder = self._build_mlp_encoder(
                config.input_dim,
                config.hidden_dims,
                config.latent_dim,
                config.dropout,
            )
            self.classical_decoder = self._build_mlp_decoder(
                config.latent_dim,
                config.hidden_dims,
                config.input_dim,
                config.dropout,
            )

        if config.mode in ("quantum", "hybrid"):
            if QuantumCircuit is None:
                raise ImportError("Qiskit must be installed for quantum mode")
            self.quantum_encoder = self._build_quantum_autoencoder(config)
            # For simplicity the quantum decoder mirrors the encoder
            self.quantum_decoder = self.quantum_encoder

    # --------------------------------------------------------------------- #
    # Classical MLP construction helpers
    # --------------------------------------------------------------------- #
    def _build_mlp_encoder(
        self,
        input_dim: int,
        hidden_dims: Tuple[int,...],
        latent_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        return nn.Sequential(*layers)

    def _build_mlp_decoder(
        self,
        latent_dim: int,
        hidden_dims: Tuple[int,...],
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    # --------------------------------------------------------------------- #
    # Quantum autoencoder construction
    # --------------------------------------------------------------------- #
    def _build_quantum_autoencoder(self, config: AutoencoderConfig) -> SamplerQNN:
        algorithm_globals.random_seed = 42
        backend = Aer.get_backend("aer_simulator_statevector")

        def ansatz(num_qubits: int) -> QuantumCircuit:
            return RealAmplitudes(num_qubits, reps=5)

        def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(qr, cr)

            # Encode + ansatz
            circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)

            # Swap‑test domain‑wall
            circuit.barrier()
            aux = num_latent + 2 * num_trash
            circuit.h(aux)
            for i in range(num_trash):
                circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
            circuit.h(aux)

            circuit.measure(aux, cr[0])
            return circuit

        num_latent = config.latent_dim
        num_trash = 2

        # Base auto‑encoder circuit
        circuit = auto_encoder_circuit(num_latent, num_trash)

        # Prepend a self‑attention block (quantum)
        sa_params = np.random.rand(num_latent * 3)
        ent_params = np.random.rand(num_latent - 1)
        sa_circuit = _build_qiskit_self_attention(num_latent, sa_params, ent_params)
        circuit.compose(sa_circuit, inplace=True)

        # Wrap into a SamplerQNN – no trainable input params, all parameters are weights
        qnn = SamplerQNN(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=(1,),
            sampler=backend,
        )
        return qnn

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def encode(
        self,
        x: torch.Tensor,
        rotation_params: Optional[torch.Tensor] = None,
        entangle_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.mode == "classical":
            return self.classical_encoder(x)
        elif self.config.mode == "quantum":
            # Quantum encoder returns a numpy array
            q_result = self.quantum_encoder.predict([])
            return torch.tensor(q_result, dtype=torch.float32)
        else:  # hybrid
            if rotation_params is None or entangle_params is None:
                raise ValueError("Hybrid mode requires rotation and entangle params")
            ca = self.classical_encoder(x)
            qa = torch.tensor(
                self.quantum_encoder.predict([]), dtype=torch.float32
            )
            return ca + qa

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.mode == "classical":
            return self.classical_decoder(z)
        else:
            # Quantum decoder is essentially the same circuit; for demo purposes
            # we just pass through the classical decoder.
            return self.classical_decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: Optional[torch.Tensor] = None,
        entangle_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encode(x, rotation_params, entangle_params)
        return self.decode(z)


# --------------------------------------------------------------------------- #
# Public factory helper
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    mode: str = "hybrid",
) -> AutoencoderNet:
    """
    Factory that returns a configured AutoencoderNet instance.

    Args:
        input_dim: Dimensionality of the input features.
        latent_dim: Size of the latent space.
        hidden_dims: Tuple of hidden layer sizes for the MLP.
        dropout: Drop‑out probability.
        mode: One of 'classical', 'quantum', or 'hybrid'.

    Returns:
        An instance of :class:`AutoencoderNet`.
    """
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        mode=mode,
    )
    return AutoencoderNet(cfg)


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "ClassicalSelfAttention",
    "ClassicalSelfAttention",
]
