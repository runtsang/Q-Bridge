"""Quantum encoder module used by the hybrid autoencoder.

Provides a variational circuit that maps an 8‑dimensional classical
feature vector to a latent vector of arbitrary dimension via
parameter‑shift gradients.  The implementation is fully compatible
with PyTorch autograd thanks to ``HybridQuantumFunction``.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli
from qiskit.providers import Backend

class QuantumEncoderCircuit:
    """Variational circuit that receives a classical input vector and
    returns expectation values of Pauli‑Z on each qubit.
    """
    def __init__(self, num_qubits: int, backend: Backend, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.backend = backend
        self.shots = shots

        # Build a depth‑1 RealAmplitudes ansatz with the required number of parameters
        self.circuit = QuantumCircuit(num_qubits)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        self.circuit.append(ansatz.decompose(), range(num_qubits))
        self.circuit.measure_all()

        self.params = ansatz.parameters
        self.sampler = StatevectorSampler(backend=self.backend)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Execute the circuit for each row of ``inputs`` and return
        the vector of Z‑expectation values.

        Parameters
        ----------
        inputs: np.ndarray
            Shape (batch, num_params).  ``num_params`` must equal
            ``len(self.params)``.
        """
        if inputs.ndim == 1:
            inputs = inputs[np.newaxis, :]
        batch = []
        for row in inputs:
            bind = {p: val for p, val in zip(self.params, row)}
            circ_copy = self.circuit.copy()
            circ_copy.assign_parameters(bind, inplace=True)
            batch.append(circ_copy)

        compiled = transpile(batch, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.sampler.run(qobj)
        result = job.result()
        # Extract statevectors
        statevecs = result.get_statevectors()
        expectations = []
        for sv in statevecs:
            exp = [np.real(sv.expectation_value(Pauli('Z').tensor_power(k))) for k in range(self.num_qubits)]
            expectations.append(exp)
        return np.array(expectations)

class HybridQuantumFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_encoder: QuantumEncoderCircuit, shift: float) -> torch.Tensor:
        ctx.quantum_encoder = quantum_encoder
        ctx.shift = shift
        np_inputs = inputs.detach().cpu().numpy()
        latent_np = quantum_encoder.run(np_inputs)
        result = torch.tensor(latent_np, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        np_inputs = inputs.detach().cpu().numpy()
        grads = []
        for i in range(inputs.shape[1]):
            plus = np_inputs.copy()
            minus = np_inputs.copy()
            plus[:, i] += shift
            minus[:, i] -= shift
            exp_plus = ctx.quantum_encoder.run(plus)
            exp_minus = ctx.quantum_encoder.run(minus)
            grad = (exp_plus - exp_minus) / 2.0
            grads.append(grad)
        grads = np.stack(grads, axis=1)  # shape (batch, params)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class HybridQuantumEncoder(nn.Module):
    """Module that exposes the quantum encoder as a PyTorch layer."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        backend: Backend,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.quantum_encoder = QuantumEncoderCircuit(num_qubits, backend, shots)
        self.shift = shift
        self.latent_dim = latent_dim
        if latent_dim!= num_qubits:
            self.project = nn.Linear(num_qubits, latent_dim)
        else:
            self.project = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = HybridQuantumFunction.apply(inputs, self.quantum_encoder, self.shift)
        if self.project:
            latent = self.project(latent)
        return latent

__all__ = [
    "QuantumEncoderCircuit",
    "HybridQuantumFunction",
    "HybridQuantumEncoder",
]
