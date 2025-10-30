"""Quantum‑only auto‑encoder implemented with Qiskit.

The class ``UnifiedAutoEncoder`` here provides a variational encoder
and decoder that operate on quantum statevectors.  It is designed to
be used as a drop‑in replacement for the classical version when a
pure‑quantum workflow is desired.  The implementation relies on
Qiskit Aer for simulation and uses a RealAmplitudes ansatz for both
the encoder and the decoder.  The readout is a simple measurement of
all qubits in the computational basis, and the loss is the squared
difference between input and reconstructed statevectors.

Author: gpt-oss-20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn

# Qiskit imports
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector


@dataclass
class UnifiedAutoEncoderConfig:
    """Configuration for the quantum auto‑encoder."""
    n_qubits: int = 4
    latent_dim: int = 4
    encoder_reps: int = 3
    decoder_reps: int = 3
    backend: str = "statevector_simulator"
    shots: int = 1


class UnifiedAutoEncoder(nn.Module):
    """Quantum auto‑encoder with encoder and decoder ansatzes."""
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_circuit = RealAmplitudes(cfg.n_qubits, reps=cfg.encoder_reps)
        self.decoder_circuit = RealAmplitudes(cfg.n_qubits, reps=cfg.decoder_reps)

        # Parameter tensors for the ansatzes
        self.encoder_params = nn.Parameter(torch.randn(self.encoder_circuit.num_parameters))
        self.decoder_params = nn.Parameter(torch.randn(self.decoder_circuit.num_parameters))

        # Backend for simulation
        self.backend = Aer.get_backend(cfg.backend)

    def _apply_circuit(self, circuit: QuantumCircuit, params: torch.Tensor, state: np.ndarray) -> np.ndarray:
        """Apply a parameterised circuit to a statevector."""
        qc = circuit.copy()
        qc.initialize(state, qc.qubits)
        qc.assign_parameters({f"theta_{i}": float(params[i].item()) for i in range(len(params))}, inplace=True)
        job = execute(qc, self.backend, shots=self.cfg.shots)
        result = job.result()
        return result.get_statevector(qc)

    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        """
        input_state: tensor of shape (batch, 2**n_qubits) – statevector amplitudes.
        Returns reconstructed statevector of the same shape.
        """
        batch = input_state.shape[0]
        reconstructed = []
        for i in range(batch):
            state = input_state[i].detach().cpu().numpy()
            # Encode
            latent = self._apply_circuit(self.encoder_circuit, self.encoder_params, state)
            # Decode
            recon = self._apply_circuit(self.decoder_circuit, self.decoder_params, latent)
            reconstructed.append(torch.tensor(recon, dtype=torch.float32))
        return torch.stack(reconstructed, dim=0)

    def loss(self, input_state: torch.Tensor, recon_state: torch.Tensor) -> torch.Tensor:
        """Mean squared error between input and reconstructed statevectors."""
        return torch.mean((input_state - recon_state) ** 2)

    def parameters(self):
        return [self.encoder_params, self.decoder_params]


__all__ = ["UnifiedAutoEncoderConfig", "UnifiedAutoEncoder"]
