"""Quantum Sampler Network using a parameterised variational circuit.

The implementation follows the original structure but enriches it with:
* A two‑qubit circuit that supports separate input and weight parameters.
* A QNode that returns the full probability distribution over the
  computational basis, enabling gradient‑based optimisation via
  Pennylane's autograd interface.
* Convenience methods to sample measurement outcomes and to
  initialise trainable parameters.

The class is designed to be drop‑in compatible with the classical
`SamplerQNN` interface: `forward` accepts an input tensor and
optionally a weight vector and returns a probability tensor.
"""

import pennylane as qml
import pennylane.numpy as np
import torch


# Device used for simulation; can be swapped for a real backend.
dev = qml.device("default.qubit", wires=2)


class SamplerQNN:
    """
    Variational quantum sampler with a 2‑qubit circuit.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input parameters (one per qubit).
    weight_dim : int, default 4
        Number of trainable weight parameters.
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 4) -> None:
        self.input_dim = input_dim
        self.weight_dim = weight_dim

        # Create placeholders for parameters that will be updated by optimisers.
        self.input_params = np.array([0.0] * input_dim, requires_grad=False)
        self.weight_params = np.array([0.0] * weight_dim, requires_grad=True)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Parameterised rotation layer
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            # Return probabilities of computational basis states
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def forward(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the probability distribution over the 2‑qubit basis.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (..., 2) – input parameters for RY gates.
        weights : torch.Tensor, optional
            Shape (..., 4) – trainable weight parameters.
            If None, the stored `self.weight_params` are used.

        Returns
        -------
        torch.Tensor
            Probability tensor of shape (..., 4).
        """
        if weights is None:
            weights = torch.tensor(self.weight_params, dtype=torch.float32)
        probs = self.circuit(inputs, weights)
        return probs

    def sample(
        self,
        inputs: torch.Tensor,
        num_shots: int = 1024,
    ) -> torch.Tensor:
        """
        Draw samples from the distribution defined by the circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Input parameters.
        num_shots : int
            Number of measurement shots per input.

        Returns
        -------
        torch.Tensor
            Sampled bit strings of shape (..., num_shots, 2) with values 0/1.
        """
        probs = self.forward(inputs)
        # Convert probabilities to a cumulative distribution for sampling.
        cum_probs = torch.cumsum(probs, dim=-1)
        rng = torch.rand(*inputs.shape[:-1], num_shots)
        # Find the first index where rng <= cum_probs
        samples = torch.argmax((rng[..., None] <= cum_probs).int(), dim=-1)
        # Convert indices to bit strings
        bit_strings = ((samples[:, :, None] & (1 << torch.arange(2))) > 0).int()
        return bit_strings

    def init_weights(self, seed: int | None = None) -> None:
        """
        Randomly initialise the trainable weights from a normal distribution.
        """
        rng = np.random.default_rng(seed)
        self.weight_params = rng.standard_normal(self.weight_dim) * 0.01


__all__ = ["SamplerQNN"]
