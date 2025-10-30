"""Quantum neural network with a 2‑qubit variational circuit.

The circuit consists of alternating layers of single‑qubit rotations
and CNOT entanglement, parameterized by input data and trainable weights.
The expectation value of Pauli‑Z on the first qubit is returned as the
regression output.  The implementation is fully differentiable in
Pennylane and can be wrapped into a PyTorch module if desired.
"""

import pennylane as qml
import torch
from torch import nn

dev = qml.device("default.qubit", wires=2)


def _variational_circuit(x: torch.Tensor, w: torch.Tensor) -> float:
    """Parameterised circuit used by the QNN.

    Parameters
    ----------
    x : torch.Tensor
        2‑dimensional input data, used as rotation angles.
    w : torch.Tensor
        Trainable weight parameters, one per rotation gate.
    """
    # Input rotations
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # First layer of trainable rotations
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)

    # Entanglement
    qml.CNOT(wires=[0, 1])

    # Second layer of trainable rotations
    qml.RX(w[2], wires=0)
    qml.RX(w[3], wires=1)

    # Entanglement
    qml.CNOT(wires=[1, 0])

    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch")
def qnn_circuit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _variational_circuit(x, w)


class EstimatorQNN(nn.Module):
    """A PyTorch‑compatible wrapper around the Pennylane QNN.

    The wrapper exposes a trainable ``weights`` tensor that is applied
    to the variational circuit.  During the forward pass the input
    features are forwarded to the circuit, and the expectation value
    of Pauli‑Z on wire 0 is returned as a scalar output.
    """

    def __init__(self) -> None:
        super().__init__()
        # 4 trainable parameters: one for each RX gate in the circuit
        self.weights = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x has shape (batch, 2)
        return qnn_circuit(x, self.weights)

__all__ = ["EstimatorQNN"]
