"""Quantum refinement of a latent vector using a RealAmplitudes circuit.

The circuit encodes each latent dimension as a rotation angle on a dedicated
qubit.  The output is the expectation value of the Pauli‑Z operator on each
qubit, producing a real‑valued vector of the same dimensionality.  This
module can be used as a drop‑in refinement step for hybrid autoencoders.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as EstimatorPrimitive
from qiskit_machine_learning.utils import algorithm_globals

class QuantumLatentRefine:
    def __init__(self, latent_dim: int, reps: int = 1, backend: str = "statevector_simulator") -> None:
        self.latent_dim = latent_dim
        self.reps = reps
        algorithm_globals.random_seed = 42
        # Build the parameterised circuit
        self.circuit = QuantumCircuit(latent_dim)
        self.circuit.append(RealAmplitudes(latent_dim, reps=reps), range(latent_dim))
        # Observables: Pauli‑Z on each qubit
        observables = []
        for i in range(latent_dim):
            pauli_str = "I" * i + "Z" + "I" * (latent_dim - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1)]))
        # Estimator QNN
        estimator = EstimatorPrimitive(backend=backend)
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observables,
            input_params=[],
            weight_params=list(self.circuit.parameters),
            estimator=estimator,
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim()!= 2 or latent.size(1)!= self.latent_dim:
            raise ValueError(f"Expected latent of shape (batch, {self.latent_dim})")
        batch_size = latent.size(0)
        refined = torch.empty_like(latent)
        for idx in range(batch_size):
            latent_np = latent[idx].cpu().numpy()
            param_dict = {p: float(v) for p, v in zip(self.circuit.parameters, latent_np)}
            out_np = self.qnn.eval(param_dict)
            refined[idx] = torch.tensor(out_np, dtype=torch.float32, device=latent.device)
        return refined

__all__ = ["QuantumLatentRefine"]
