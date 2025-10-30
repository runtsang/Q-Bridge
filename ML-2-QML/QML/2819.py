"""Quantum‑parameterised self‑attention block.

The quantum implementation follows the interface of the classical
`SelfAttentionHybrid`.  It generates rotation and entanglement angles
via a Pennylane variational circuit and then applies the same
classical attention formula.  The module is fully differentiable
and can be trained end‑to‑end with PyTorch optimisers.

Key ideas from the seeds:
* Variational circuit that produces rotation and entanglement
  parameters (analogous to the QLSTM quantum gates).
* Use of a small quantum LSTM‑style cell to modulate attention
  scores, providing a hybrid gating mechanism.
* Compatibility with the seed interface, so the class can be
  swapped in without changing downstream code.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
import torch.nn as nn

class SelfAttentionHybrid:
    """
    Quantum‑parameterised self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode each feature vector.
    embed_dim : int
        Dimensionality of the input embeddings.
    """

    def __init__(self, n_qubits: int, embed_dim: int) -> None:
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        # Device with enough qubits to handle the feature vector
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Learnable parameters for the variational circuit
        self.params = nn.Parameter(
            torch.randn(n_qubits * 2)  # rotation + entanglement
        )

        # Small quantum LSTM‑style cell implemented as a circuit
        # that produces a gating vector for attention scores.
        self.lstm_params = nn.Parameter(
            torch.randn(n_qubits * 4)  # four gates per qubit
        )

    def _variational_circuit(self, inputs: torch.Tensor, params: torch.Tensor):
        """
        Variational circuit that encodes the input and produces
        rotation and entanglement angles as expectation values.
        """
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inp, param):
            # Encode each feature into a separate qubit via RX
            for i, x in enumerate(inp):
                qml.RX(x, wires=i)

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(param[i], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(param[self.n_qubits + i], wires=i + 1)

            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(inputs, params)

    def _lstm_gate(self, inputs: torch.Tensor, gate_params: torch.Tensor):
        """
        Quantum LSTM‑style gate that outputs a gating vector.
        The gate is implemented as a small circuit that produces
        four expectation values per qubit (forget, input, update, output).
        """
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def gate_circuit(inp, gp):
            # Encode input
            for i, x in enumerate(inp):
                qml.RX(x, wires=i)

            # Apply four parameterised rotations per qubit
            for i in range(self.n_qubits):
                idx = 4 * i
                qml.RX(gp[idx], wires=i)       # forget
                qml.RY(gp[idx + 1], wires=i)   # input
                qml.RZ(gp[idx + 2], wires=i)   # update
                qml.RX(gp[idx + 3], wires=i)   # output

            # Entangle qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Return expectation values of PauliZ (one per qubit)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # The circuit returns only one expectation per qubit; for simplicity
        # we replicate it for all four gates (this is a placeholder for a
        # more elaborate implementation).  In practice, one would measure
        # different observables for each gate.
        z_vals = gate_circuit(inputs, gate_params)
        # Repeat each value four times to emulate four gates
        gate_vec = torch.stack([z_vals for _ in range(4)], dim=0).reshape(-1)
        return gate_vec  # (4 * n_qubits,)

    def run(
        self,
        inputs: torch.Tensor,
        shots: int = 1024,
    ) -> torch.Tensor:
        """
        Forward pass of the quantum self‑attention block.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).
        shots : int, default 1024
            Number of shots for the simulation (unused with
            Pennylane's analytic backend).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape
            (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Flatten batch and sequence for circuit evaluation
        flat_inputs = inputs.reshape(-1, self.embed_dim)

        # Generate rotation parameters for each input vector
        rot_params = []
        for inp in flat_inputs:
            out = self._variational_circuit(inp, self.params)
            rot_params.append(out)
        rot_params = torch.stack(rot_params).reshape(batch, seq_len, self.n_qubits)

        # Use rotation params as key; query and value remain the raw inputs
        query = inputs
        key = rot_params
        value = inputs

        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.embed_dim)
        scores = torch.softmax(scores, dim=-1)

        # Quantum LSTM‑style gating
        gate_vecs = []
        for inp in flat_inputs:
            gv = self._lstm_gate(inp, self.lstm_params)
            gate_vecs.append(gv)
        gate_vecs = torch.stack(gate_vecs).reshape(batch, seq_len, self.n_qubits * 4)

        # Reshape gate vector into four gates per qubit
        f = torch.sigmoid(gate_vecs[:, :, : self.n_qubits])
        i = torch.sigmoid(gate_vecs[:, :, self.n_qubits : 2 * self.n_qubits])
        g = torch.tanh(gate_vecs[:, :, 2 * self.n_qubits : 3 * self.n_qubits])
        o = torch.sigmoid(gate_vecs[:, :, 3 * self.n_qubits :])

        # Modulate scores with LSTM‑style gate
        gate = o * torch.tanh(f * g)
        scores = scores * gate

        # Weighted sum
        out = torch.bmm(scores, value)
        return out

__all__ = ["SelfAttentionHybrid"]
