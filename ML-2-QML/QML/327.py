"""Quantum regression model using a Pennylane variational circuit."""

import pennylane as qml
import torch
from torch import nn
import numpy as np

# Three‑qubit device; the third qubit holds the trainable weight.
dev = qml.device("default.qubit", wires=3)


def _quantum_circuit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Variational circuit that encodes a 2‑D input `x` and a single weight `w`.

    Parameters
    ----------
    x : torch.Tensor
        Input vector of shape (2,). Encoded via RY rotations on qubits 0 and 1.
    w : torch.Tensor
        Trainable weight scalar. Encoded via RX on qubit 2.

    Returns
    -------
    torch.Tensor
        Expectation value <Y> on qubit 2.
    """
    # Encode inputs
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # Encode weight
    qml.RX(w, wires=2)

    # Entangle the weight qubit with the input qubits
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])

    # Measure Pauli‑Y on the weight qubit
    return qml.expval(qml.PauliY(2))


class EstimatorQNN(nn.Module):
    """
    Hybrid quantum‑classical regressor.

    The network exposes a single trainable parameter `weight` that is fed into the quantum circuit.
    Inputs are expected to be 2‑dimensional feature vectors. The output is the expectation value
    of Pauli‑Y on the third qubit, which serves as the regression prediction.

    The circuit can be extended with additional variational layers or observables
    without altering the public interface.
    """
    def __init__(self, weight_init: float = 0.0) -> None:
        super().__init__()
        # Trainable weight parameter
        self.weight = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32))
        # Wrap the circuit as a Pennylane QNode with Torch interface
        self._qnode = qml.QNode(_quantum_circuit, dev, interface="torch")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that runs the quantum circuit for each input in the batch.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1), containing expectation values.
        """
        # The QNode automatically maps over the first dimension of the input.
        return self._qnode(inputs, self.weight)


__all__ = ["EstimatorQNN"]
