"""Quantum regression estimator based on PennyLane.

The module implements a variational circuit with a trainable
parameterised ansatz.  The circuit is wrapped in a PennyLane
QNode that returns a scalar expectation value for regression.
The public API mirrors the original seed: calling EstimatorQNN()
produces an object that can be used as a PyTorch module via
the ``forward`` method.
"""

import pennylane as qml
import torch
from typing import Optional, Sequence

# Default device: use a GPU‑capable backend if available
dev = qml.device("default.qubit", wires=2)

class PennylaneEstimatorQNN:
    """
    Quantum neural network implemented with PennyLane.

    Parameters
    ----------
    depth : int, default 2
        Number of variational layers.
    entanglement : str, default "full"
        Entanglement scheme for each layer.
    seed : Optional[int]
        Random seed for reproducibility.
    """
    def __init__(
        self,
        depth: int = 2,
        entanglement: str = "full",
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)

        self.depth = depth
        self.entanglement = entanglement

        # Parameters: shape (depth, 2, 3) for Rot gates on each qubit
        self.params = torch.nn.Parameter(
            torch.rand((depth, 2, 3), dtype=torch.float64)
        )

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode inputs
            qml.RX(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Variational layers
            for layer in range(depth):
                for qubit in range(2):
                    qml.Rot(params[layer, qubit, 0],
                            params[layer, qubit, 1],
                            params[layer, qubit, 2],
                            wires=qubit)
                # Entanglement
                if self.entanglement == "full":
                    qml.CNOT(wires=[0, 1])
                else:
                    qml.CNOT(wires=[0, 1])  # simple two‑qubit CNOT

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) containing two real features.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) with the expectation value.
        """
        batch_size = inputs.shape[0]
        # Vectorise the circuit for batch processing
        outputs = torch.stack(
            [self.circuit(inp, self.params) for inp in inputs]
        )
        return outputs.unsqueeze(-1)

def EstimatorQNN(
    depth: int = 2,
    entanglement: str = "full",
    seed: Optional[int] = None,
) -> PennylaneEstimatorQNN:
    """
    Factory that returns a quantum estimator ready for training.

    Parameters
    ----------
    depth : int
        Number of variational layers.
    entanglement : str
        Entanglement scheme.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    PennylaneEstimatorQNN
        Instantiated quantum estimator.
    """
    return PennylaneEstimatorQNN(depth, entanglement, seed)

__all__ = ["EstimatorQNN"]
