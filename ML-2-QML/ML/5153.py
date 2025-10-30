from __future__ import annotations

import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

class HybridAutoencoderNet(nn.Module):
    """
    Classical MLP encoder/decoder with an optional quantum latent layer.
    The quantum block uses a RealAmplitudes ansatz and a Qiskit sampler
    to map classical latent codes to quantum states.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum

        # Classical encoder
        enc_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Quantum latent layer
        if use_quantum:
            self.sampler = Sampler()
            self.num_qubits = latent_dim
            self.ansatz = RealAmplitudes(self.num_qubits, reps=3)

        # Classical decoder
        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space, optionally via a quantum circuit."""
        latent = self.encoder(x)
        if self.use_quantum:
            # Map latent vector to circuit parameters
            params = latent.detach().cpu().numpy()
            qc = QuantumCircuit(self.num_qubits)
            qc.append(self.ansatz, range(self.num_qubits))
            qc.set_parameters(params)
            result = self.sampler.run(qc, shots=1).result()
            state = Statevector(result.get_statevector())
            # Use real part of the state as a deterministic embedding
            latent_q = torch.tensor(state.data.real[: self.num_qubits], device=x.device, dtype=torch.float32)
            return latent_q
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))
