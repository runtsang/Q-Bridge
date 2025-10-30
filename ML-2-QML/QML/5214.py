"""Hybrid LSTM that uses Qiskit variational circuits for the gates.

The implementation mirrors :class:`QLSTMHybrid` from the classical
module but replaces each gate with a small Qiskit circuit.  Each
gate is a parameterised RealAmplitudes ansatz followed by a chain of
CNOTs.  The circuit is evaluated with an EstimatorQNN, returning the
expectation value of Pauli‑Z on the last qubit.  The API is identical
to the classical version so that the two modules can be used
interchangeably.

"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Qiskit imports ----------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

# ----------------------------------------------------------------------
#  Quantum gate using EstimatorQNN
# ----------------------------------------------------------------------
class QuantumGateQiskit(nn.Module):
    """A single quantum gate implemented with a Qiskit variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the RealAmplitudes ansatz.
    """
    def __init__(self, n_qubits: int, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        # Define circuit with symbolic parameters
        self.input_params = ParameterVector("x", n_qubits)
        self.weight_params = ParameterVector("theta", n_qubits * depth)

        self.circuit = QuantumCircuit(n_qubits)
        # Input encoding
        for i in range(n_qubits):
            self.circuit.rx(self.input_params[i], i)

        # Variational ansatz
        idx = 0
        for _ in range(depth):
            for q in range(n_qubits):
                self.circuit.ry(self.weight_params[idx], q)
                idx += 1
            for q in range(n_qubits - 1):
                self.circuit.cz(q, q + 1)

        # Expectation observable
        obs = SparsePauliOp.from_list([("Z", 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[obs],
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        x_np = x.detach().cpu().numpy()
        out = self.qnn(x_np)  # shape (batch, 1)
        return torch.tensor(out, dtype=x.dtype, device=x.device)

# ----------------------------------------------------------------------
#  Hybrid LSTM (Qiskit version)
# ----------------------------------------------------------------------
class QLSTMHybridQ(nn.Module):
    """Hybrid LSTM that uses Qiskit circuits for every gate.

    The API matches :class:`QLSTMHybrid` so that the two modules can
    be swapped in the same training loop.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Quantum gates
        self.forget = QuantumGateQiskit(n_qubits, depth)
        self.input = QuantumGateQiskit(n_qubits, depth)
        self.update = QuantumGateQiskit(n_qubits, depth)
        self.output = QuantumGateQiskit(n_qubits, depth)

        # Linear projections to map classical data into qubits
        self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch, self.hidden_dim, device=device)
        cx = torch.zeros(batch, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.forget_proj(combined)))
            i = torch.sigmoid(self.input(self.input_proj(combined)))
            g = torch.tanh(self.update(self.update_proj(combined)))
            o = torch.sigmoid(self.output(self.output_proj(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

__all__ = ["QLSTMHybridQ"]
