"""Quantum EstimatorQNN__gen384 using PennyLane.

This module defines a quantum circuit that encodes an input vector via
RX rotations, applies a depth‑scaled variational ansatz of RY and CZ
gates, and measures the expectation of Z on each qubit.  The circuit
is wrapped in a torch.autograd.Function so that it can be used as a
drop‑in replacement for a classical layer.

Class
------
EstimatorQNN__gen384
    PyTorch module exposing a quantum layer.
"""

import pennylane as qml
import torch
import math
from typing import List

class _QuantumNet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params: torch.Tensor, input_vec: torch.Tensor, num_qubits: int, depth: int):
        """
        Execute a parameterised quantum circuit.

        Parameters
        ----------
        params : torch.Tensor
            Flattened vector of length ``num_qubits + num_qubits*depth``.
            The first ``num_qubits`` entries are encoding parameters (RX gates),
            the rest are variational parameters (RY gates in each layer).
        input_vec : torch.Tensor
            One‑dimensional tensor of length ``num_qubits`` containing the
            classical input that will be encoded with RX rotations.
        num_qubits : int
            Number of qubits in the ansatz.
        depth : int
            Number of variational layers.

        Returns
        -------
        torch.Tensor
            Expectation values of Z on each qubit.
        """
        enc_params = params[:num_qubits]
        weight_params = params[num_qubits:]

        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor):
            # Encoding
            for i in range(num_qubits):
                qml.RX(x[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RY(weight_params[idx], wires=i)
                    idx += 1
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        return circuit(input_vec)

class EstimatorQNN__gen384(torch.nn.Module):
    """
    Quantum layer that can be plugged into a classical network.

    Parameters
    ----------
    num_qubits : int, default 384
        Size of the quantum register.  Must match the feature dimension
        expected by the user.
    depth : int, default 3
        Depth of the variational ansatz.
    device : str, default "cpu"
        Torch device on which parameters are stored and gradients
        are computed.
    """
    def __init__(self, num_qubits: int = 384, depth: int = 3, device: str = "cpu"):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        self.num_params = num_qubits + num_qubits * depth
        # Initialise parameters with a normal distribution
        self.params = torch.nn.Parameter(
            torch.randn(self.num_params, device=self.device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, num_qubits)``.  Each row is
            treated as a separate example and is run through the circuit
            sequentially.  The method currently supports a single
            example per batch for simplicity.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, num_qubits)`` containing the
            expectation values of Z on each qubit.
        """
        if x.dim()!= 2 or x.size(1)!= self.num_qubits:
            raise ValueError(
                f"Input must have shape (batch, {self.num_qubits}) but got {x.shape}"
            )
        batch_size = x.size(0)
        out = []
        for i in range(batch_size):
            out.append(_QuantumNet.apply(self.params, x[i], self.num_qubits, self.depth))
        return torch.stack(out)

__all__ = ["EstimatorQNN__gen384"]
