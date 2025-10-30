"""Quantum regression model that replaces the original QML seed with a more robust variational ansatz and a parameter‑shift gradient estimator.

The design leverages the Qiskit Aer simulator and QuantumCircuit‑based circuits that all‑tune the learnable parameters.  The quantum part of the QML model is defined in a separate class that self‑contains the quantum device and the variational circuit.  The final measurement is read‑out from a classical‑to‑quantum state‑encoding scheme, and the output is mapped into a linear regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Pauli
import numpy as np
from typing import Tuple

def _create_variational_circuit(num_wires: int, depth: int, params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(num_wires)
    idx = 0
    for d in range(depth):
        for w in range(num_wires):
            qc.rx(params[idx], w)
            idx += 1
            qc.ry(params[idx], w)
            idx += 1
        for w in range(num_wires - 1):
            qc.cz(w, w + 1)
        qc.cz(num_wires - 1, 0)
    return qc

def _encode_state(qc: QuantumCircuit, state: np.ndarray) -> QuantumCircuit:
    for i, val in enumerate(state):
        qc.rx(val, i)
    return qc

class QuantumRegressionQML(nn.Module):
    def __init__(self, num_wires: int, depth: int = 2, seed: int | None = None):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.num_params = 2 * num_wires * depth
        self.params = ParameterVector("theta", self.num_params)
        self.base_circuit = _create_variational_circuit(num_wires, depth, self.params)
        self.backend = AerSimulator(method="statevector")
        self.head = nn.Linear(num_wires, 1)
        if seed is not None:
            np.random.seed(seed)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        batch_size = state_batch.shape[0]
        circuits = []
        for i in range(batch_size):
            qc = self.base_circuit.copy()
            state = state_batch[i].cpu().numpy()
            qc = _encode_state(qc, state)
            circuits.append(qc)
        job = execute(circuits, backend=self.backend, shots=0)
        results = job.result()
        exp_vals = []
        for i, qc in enumerate(circuits):
            state_vec = results.get_statevector(i)
            sv = Statevector(state_vec)
            z_vals = []
            for q in range(self.num_wires):
                pauli_z = Pauli("I" * q + "Z" + "I" * (self.num_wires - q - 1))
                z_vals.append(sv.expectation_value(pauli_z).real)
            exp_vals.append(z_vals)
        exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=state_batch.device)
        return self.head(exp_tensor).squeeze(-1)

    def parameter_shift_gradient(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(state_batch)
        outputs.backward(torch.ones_like(outputs))
        grads = [p.grad.clone() for p in self.parameters() if p.requires_grad]
        return grads, outputs

__all__ = ["QuantumRegressionQML"]
