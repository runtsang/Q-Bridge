"""Quantum components for the unified hybrid layer.

Defines a simple parameterized quantum encoder and a variational LSTM
that uses the encoder for its gates.  The encoder uses Qiskit to
simulate a small circuit and returns expectation values of Pauli‑Z
for each qubit.  The LSTM is a drop‑in replacement for a classical
LSTM when `use_variational_lstm` is True.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class QuantumEncoder(nn.Module):
    """
    Parameterized quantum encoder that maps an input vector of length
    `n_qubits` onto expectation values of Pauli‑Z on each qubit.
    Uses a state‑vector simulator for exact results.
    """
    def __init__(self, n_qubits: int = 4, backend=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        thetas: np.ndarray
            Shape (batch, n_qubits) – rotation angles for each qubit.

        Returns
        -------
        np.ndarray
            Shape (batch, n_qubits) – expectation values of Z.
        """
        batch_size = thetas.shape[0]
        expectations = np.zeros((batch_size, self.n_qubits))
        for i in range(batch_size):
            theta_vals = thetas[i]
            qc = QuantumCircuit(self.n_qubits)
            # Entangle all qubits with Hadamard
            qc.h(range(self.n_qubits))
            # Encode angles
            for q in range(self.n_qubits):
                qc.ry(theta_vals[q], q)
            # Obtain statevector
            result = execute(qc, self.backend).result()
            statevector = result.get_statevector(qc)
            # Compute expectation of Z for each qubit
            for q in range(self.n_qubits):
                exp = 0.0
                for idx, amp in enumerate(statevector):
                    bit = (idx >> q) & 1
                    exp += (1 if bit == 0 else -1) * abs(amp) ** 2
                expectations[i, q] = exp
        return expectations

class QuantumLSTM(nn.Module):
    """
    LSTM where each gate is realised by a quantum circuit.
    The gates are produced by passing a linear projection of the
    concatenated input and hidden state through a QuantumEncoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to n_qubits for each gate
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.quantum_encoder = QuantumEncoder(n_qubits=n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs: torch.Tensor
            Shape (seq_len, batch, input_dim)
        states: tuple
            (hidden_state, cell_state)

        Returns
        -------
        outputs: torch.Tensor
            Shape (seq_len, batch, hidden_dim)
        new_states: tuple
            Updated hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):  # iterate over sequence
            combined = torch.cat([x, hx], dim=1)  # (batch, input_dim + hidden_dim)
            # Forget gate
            f_q = self.quantum_encoder.run(
                self.forget_linear(combined).detach().cpu().numpy()
            )
            f = torch.sigmoid(torch.from_numpy(f_q).to(x.device).float())
            # Input gate
            i_q = self.quantum_encoder.run(
                self.input_linear(combined).detach().cpu().numpy()
            )
            i = torch.sigmoid(torch.from_numpy(i_q).to(x.device).float())
            # Update gate
            g_q = self.quantum_encoder.run(
                self.update_linear(combined).detach().cpu().numpy()
            )
            g = torch.tanh(torch.from_numpy(g_q).to(x.device).float())
            # Output gate
            o_q = self.quantum_encoder.run(
                self.output_linear(combined).detach().cpu().numpy()
            )
            o = torch.sigmoid(torch.from_numpy(o_q).to(x.device).float())

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

__all__ = ["QuantumEncoder", "QuantumLSTM"]
