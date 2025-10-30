import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridEstimator(nn.Module):
    """Hybrid classical‑quantum regressor that combines a classical MLP,
    a variational quantum circuit and a quantum autoencoder.  This
    follows the *combination* scaling paradigm: classical layers
    extract features while the quantum layer learns depth‑scaling
    non‑linearities.
    """

    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 16,
                 latent_dim: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Quantum regression layer
        self._quantum_params = [Parameter(f"q{i}") for i in range(3)]
        self._qc = self._build_variational_circuit()
        # Quantum autoencoder
        self._ae_params = [Parameter(f"a{i}") for i in range(latent_dim)]
        self._ae_circuit = self._build_autoencoder(latent_dim)
        self.estimator = StatevectorEstimator()

    def _build_variational_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self._quantum_params[0], 0)
        qc.rx(self._quantum_params[1], 0)
        qc.rz(self._quantum_params[2], 0)
        return qc

    def _build_autoencoder(self, latent_dim: int) -> QuantumCircuit:
        qc = QuantumCircuit(latent_dim + 1)
        for i in range(latent_dim):
            qc.rx(self._ae_params[i], i)
        for i in range(latent_dim):
            qc.cx(i, latent_dim)
        qc.measure(latent_dim, 0)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        param_bind = [{p: v.item() for p, v in zip(self._quantum_params, x.squeeze())}]
        state = self.estimator.run(self._qc, param_bind)[0]
        z_exp = state.expectation_value(SparsePauliOp.from_list([("Z", 1)])).real
        ae_bind = [{p: 0.0 for p in self._ae_params}]
        compressed = self.estimator.run(self._ae_circuit, ae_bind)[0]
        compressed_val = compressed.expectation_value(SparsePauliOp.from_list([("Z", 1)])).real
        out = x.squeeze() + torch.tensor(z_exp + compressed_val, device=x.device)
        return out

__all__ = ["HybridEstimator"]
