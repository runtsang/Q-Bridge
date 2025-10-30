"""
EstimatorQNNEnhanced: A hybrid quantum‑classical variational regressor.
Implemented with Pennylane, the circuit accepts two classical inputs and a single
trainable weight.  The observable is a Pauli‑Y operator on the single qubit.
The class provides the same forward signature as the classical counterpart,
returning the expectation value as a torch tensor for seamless loss computation.
"""

from __future__ import annotations

import pennylane as qml
import torch
from pennylane import numpy as np
from typing import Sequence, Callable, Optional


class EstimatorQNNEnhanced:
    """
    Variational quantum neural network with configurable depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Default is 1.
    depth : int
        Number of parameterized layers. Default is 2.
    observable : str | qml.operation
        Pauli observable to measure. Default is 'Y'.
    """

    def __init__(
        self,
        num_qubits: int = 1,
        depth: int = 2,
        observable: str | qml.operation = "Y",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.observable = qml.PauliY(wires=range(num_qubits)) if observable == "Y" else observable

        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Parameter registers: one input and one weight per qubit per layer
        self.input_params = [qml.Param("x") for _ in range(num_qubits)]
        self.weight_params = [qml.Param("w") for _ in range(num_qubits * depth)]

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs
            for i, param in enumerate(self.input_params):
                qml.RY(inputs[i], wires=i)
            # Parameterized layers
            weight_iter = iter(weights)
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RZ(next(weight_iter), wires=i)
                # Entangling layer
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement
            return qml.expval(self.observable)

        self.circuit = circuit

    def forward(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the expectation value of the observable.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (num_qubits,) containing the classical inputs.
        weights : torch.Tensor, optional
            Vector of trainable weights. If None, the circuit is evaluated with
            zero weights.

        Returns
        -------
        torch.Tensor
            Expectation value as a scalar tensor.
        """
        if weights is None:
            weights = torch.zeros(self.num_qubits * self.depth)
        return self.circuit(inputs, weights)


def EstimatorQNNEnhancedModel() -> EstimatorQNNEnhanced:
    """Convenience factory returning a default‑configured quantum model."""
    return EstimatorQNNEnhanced()


__all__ = ["EstimatorQNNEnhanced", "EstimatorQNNEnhancedModel"]
