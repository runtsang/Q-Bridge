"""Advanced quantum sampler network.

Implemented as a Pennylane variational circuit that takes two input
angles and four trainable weights, then returns a probability
distribution over two measurement outcomes.  The circuit can be
sampled, differentiated with the parameter‑shift rule, and used
within hybrid training loops."""
from __future__ import annotations

import pennylane as qml
import torch
from pennylane import numpy as np
from typing import Tuple


class AdvancedSamplerQNN:
    """
    Variational quantum circuit with 2 qubits that mimics the
    classical sampler interface.

    Parameters
    ----------
    device : str or pennylane.Device
        Quantum device to run the circuit on.
    init_weights : Tuple[float, float, float, float], optional
        Initial values for the 4 trainable rotation angles.
    """

    def __init__(
        self,
        device: str | qml.Device = "default.qubit",
        init_weights: Tuple[float, float, float, float] | None = None,
    ) -> None:
        self.dev = qml.device(device, wires=2)
        if init_weights is None:
            init_weights = (0.0, 0.0, 0.0, 0.0)
        self.params = qml.numpy.array(init_weights, requires_grad=True)

    def circuit(self, inputs: Tuple[float, float]) -> Tuple[float, float]:
        """Parameterized quantum circuit."""
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(self.params[0], wires=0)
        qml.RY(self.params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(self.params[2], wires=0)
        qml.RY(self.params[3], wires=1)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute probability distribution for each input.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) containing rotation angles.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 2) with softmax probabilities.
        """
        probs = []
        for inp in inputs:
            expect = self.dev.batch_execute(
                [qml.QNode(self.circuit, self.dev)(tuple(inp)) for _ in range(1)]
            )[0]
            # Convert expectation values to probabilities via
            # (1 + expval) / 2 for each qubit.
            prob = (1 + np.array(expect)) / 2
            probs.append(prob)
        probs = np.stack(probs, axis=0)
        probs = qml.math.softmax(probs, axis=-1)
        return torch.tensor(probs, dtype=torch.float32)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Generate samples by measuring the circuit repeatedly.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) containing input angles.
        n_samples : int
            Number of measurements per input.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, n_samples, 2) containing one‑hot
            encoded samples of measurement outcomes.
        """
        batch = inputs.shape[0]
        all_samples = []
        for idx in range(batch):
            meas = self.dev.measure(
                [qml.QNode(self.circuit, self.dev)(tuple(inputs[idx])) for _ in range(n_samples)]
            )
            # Convert measurement bits to one‑hot encoding
            samples = torch.tensor(meas, dtype=torch.long)
            samples = torch.nn.functional.one_hot(samples, num_classes=2).float()
            all_samples.append(samples)
        return torch.stack(all_samples, dim=0)

    def parameters(self):
        """Return the trainable parameters as a torch tensor."""
        return torch.tensor(self.params, dtype=torch.float32, requires_grad=True)

    def set_parameters(self, new_params: torch.Tensor) -> None:
        """Set new parameters for the circuit."""
        self.params = new_params.detach().numpy()


__all__ = ["AdvancedSamplerQNN"]
