"""Quantum estimator using a variational circuit with Pennylane."""
from __future__ import annotations

import pennylane as qml
import numpy as np
import torch

__all__ = ["EstimatorQNNGen"]


class EstimatorQNNGen:
    """
    Variational quantum neural network that maps 2‑dimensional classical inputs
    to a continuous output via the expectation value of the Z operator on the
    first qubit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (default 2).
    n_layers : int
        Number of variational layers (default 2).
    dev : qml.Device | str | None
        Pennylane device. If None, defaults to the local `default.qubit` simulator.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        dev: qml.Device | str | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        # Parameters: input (2) + weight (n_layers * n_qubits * 2)
        self.input_params = [qml.Param("x1"), qml.Param("x2")]
        self.weight_params = qml.numpy.ndarray(
            shape=(n_layers, n_qubits, 2), requires_grad=True
        )

        # Build qnode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs via RY rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Variational layers
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                # Entanglement
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the quantum circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch_size, 2) with input features.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 1) with predicted values.
        """
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.circuit(inputs[i], self.weight_params)
            outputs.append(out)
        return torch.stack(outputs).unsqueeze(-1)

    def predict(self, inputs: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Convenience method for inference.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data of shape (batch_size, 2).
        device : torch.device or str
            Device to perform computation on (ignored for Pennylane).

        Returns
        -------
        torch.Tensor
            Predicted continuous outputs.
        """
        self.circuit.device = self.dev  # ensure device consistency
        return self.forward(inputs.to(torch.float32))

    def gradient(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients of the output w.r.t. the weight parameters
        using Pennylane's autograd.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Gradients of shape (n_layers, n_qubits, 2).
        """
        return torch.autograd.grad(
            outputs=self.forward(inputs).sum(),
            inputs=self.weight_params,
            create_graph=False,
            retain_graph=False,
        )[0]
