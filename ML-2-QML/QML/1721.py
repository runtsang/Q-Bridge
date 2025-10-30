"""QML implementation of SamplerQNN using PennyLane.

The class builds a hybrid variational circuit with the same input dimension
as the classical model.  It provides:
  * A PennyLane QNode that accepts classical inputs and weight parameters.
  * Evaluation on a chosen quantum backend (default state‑vector simulator).
  * Utility to convert the QNode to a classical equivalent by extracting the
    weight tensors.
"""

from __future__ import annotations

import pennylane as qml
import torch
from typing import Tuple, Dict

__all__ = ["SamplerQNN"]


class SamplerQNN:
    """
    Hybrid quantum sampler.

    Parameters
    ----------
    input_dim : int
        Number of classical input features (default 2).
    num_qubits : int
        Number of qubits in the circuit (default 2).
    num_layers : int
        Depth of the variational ansatz (default 2).
    device : str
        PennyLane device to run the circuit on (default "default.qubit").
    seed : Optional[int]
        Random seed for weight initialization.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_qubits: int = 2,
        num_layers: int = 2,
        device: str = "default.qubit",
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = qml.device(device, wires=num_qubits, shots=1024)
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize weight parameters
        self.weight_params = torch.nn.Parameter(
            torch.randn(num_layers * num_qubits * 3) * 0.1
        )

        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Classical input of shape (input_dim,).
        weights : torch.Tensor
            Weight parameters of shape (num_layers * num_qubits * 3,).

        Returns
        -------
        torch.Tensor
            Probability vector over basis states.
        """
        idx = 0
        # Encode inputs with Ry rotations
        for i in range(self.input_dim):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for _ in range(self.num_layers):
            for q in range(self.num_qubits):
                qml.RY(weights[idx], wires=q)
                idx += 1
                qml.RZ(weights[idx], wires=q)
                idx += 1
                qml.RX(weights[idx], wires=q)
                idx += 1
            # Entangling layer
            for q in range(self.num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        # Measure probabilities
        return qml.probs(wires=range(self.num_qubits))

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the output probability distribution for a batch of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 2**num_qubits) with probabilities.
        """
        batch = inputs.shape[0]
        probs = torch.stack(
            [self.qnode(inp, self.weight_params) for inp in inputs]
        )
        return probs

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the quantum sampler.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, input_dim).
        n_samples : int
            Number of samples per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (batch, n_samples).
        """
        probs = self.evaluate(inputs)
        return torch.multinomial(probs, n_samples, replacement=True)

    def export_weights(self) -> Dict[str, torch.Tensor]:
        """
        Export the weight parameters as a dictionary for classical use.

        Returns
        -------
        dict
            Mapping from parameter name to torch.Tensor.
        """
        return {"weight_params": self.weight_params.detach().clone()}
