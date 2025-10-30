"""Quantum head for the HybridQuantumClassifier.

This module implements a data‑re‑uploading variational circuit that
accepts a batch of qubit angles and returns the expectation value of
the Pauli‑Z operator on the first qubit.  The circuit is executed on
the Aer simulator and wrapped in a simple autograd function that
provides gradients via the parameter‑shift rule.

The design extends the quantum circuit from ``QuantumClassifierModel.py``
and the hybrid head from ``ClassicalQuantumBinaryClassification.py``,
introducing an explicit data‑re‑uploading ansatz and finite‑difference
gradient support.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from typing import List

class QuantumExpectationHead(nn.Module):
    """
    Quantum head that maps a vector of angles to a scalar expectation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    shots : int, default=1024
        Number of shots per evaluation.
    shift : float, default=π/2
        Shift used in the parameter‑shift rule for gradients.
    """
    def __init__(self,
                 n_qubits: int,
                 shots: int = 1024,
                 shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift

        # Base circuit with parameter placeholders
        self.circuit = QuantumCircuit(n_qubits)
        # Encoding: RX on each qubit
        for q in range(n_qubits):
            self.circuit.rx(qiskit.circuit.Parameter(f'x_{q}'), q)
        # Two layers of data‑re‑uploading ansatz
        for _ in range(2):
            for q in range(n_qubits):
                self.circuit.ry(qiskit.circuit.Parameter(f'theta_{q}'), q)
            for q in range(n_qubits - 1):
                self.circuit.cz(q, q + 1)
        self.circuit.measure_all()

        self.backend = AerSimulator()
        self.compiled_circ = transpile(self.circuit, self.backend)

    def _expectation(self, angle_batch: np.ndarray) -> np.ndarray:
        """Compute expectation of Z on the first qubit for each input."""
        batch_size = angle_batch.shape[0]
        param_binds: List[dict] = []
        for i in range(batch_size):
            bind = {}
            for q in range(self.n_qubits):
                bind[self.circuit.parameters[f'x_{q}']] = angle_batch[i, q]
                bind[self.circuit.parameters[f'theta_{q}']] = angle_batch[i, q]
            param_binds.append(bind)
        qobj = assemble(self.compiled_circ,
                        shots=self.shots,
                        parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts_list = result.get_counts()

        expectations: List[float] = []
        for counts in counts_list:
            exp = 0.0
            total = 0
            for bitstring, c in counts.items():
                val = 1.0 if bitstring[-1] == '0' else -1.0
                exp += val * c
                total += c
            expectations.append(exp / total)
        return np.array(expectations, dtype=np.float32)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with finite‑difference gradient support.

        Parameters
        ----------
        angles : torch.Tensor
            Tensor of shape (batch, n_qubits) containing rotation angles.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) with expectation values.
        """
        ang_np = angles.detach().cpu().numpy()
        exp = self._expectation(ang_np)
        exp_tensor = torch.tensor(exp,
                                  device=angles.device,
                                  dtype=torch.float32)

        # Simple finite‑difference gradient support
        class _ExpectFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp: torch.Tensor, head):
                ctx.head = head
                ctx.save_for_backward(inp)
                return head(inp)

            @staticmethod
            def backward(ctx, grad_output):
                inp, = ctx.saved_tensors
                shift = ctx.head.shift
                batch, dim = inp.shape
                grads = torch.zeros_like(inp)
                for i in range(batch):
                    for j in range(dim):
                        plus = inp.clone()
                        minus = inp.clone()
                        plus[i, j] += shift
                        minus[i, j] -= shift
                        e_plus = ctx.head(plus[i:i+1])
                        e_minus = ctx.head(minus[i:i+1])
                        grads[i, j] = (e_plus - e_minus) / (2 * np.sin(shift))
                return grads * grad_output, None

        return _ExpectFn.apply(exp_tensor, self)

__all__ = ["QuantumExpectationHead"]
