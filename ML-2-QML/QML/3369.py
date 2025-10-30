import torch
import torchquantum as tq
import numpy as np

class SamplerQNNGen(tq.QuantumModule):
    """
    Quantum sampler that prepares a two‑qubit state using a
    parameterized circuit driven by classical input and weight vectors.
    The circuit mirrors the classical network's output space and
    produces a probability distribution over the computational basis.
    """

    def __init__(self, n_wires: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Execute the parameterized circuit and return the probability
        distribution of measuring the |00⟩ and |01⟩ basis states.

        Parameters
        ----------
        x : torch.Tensor
            Classical input of shape (2,).
        w : torch.Tensor
            Weight vector of shape (4,) produced by the classical SamplerQNNGen.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (2,) corresponding to the |00⟩ and |01⟩ outcomes.
        """
        # Reset the device to a fresh state
        self.q_device.reset_states(1)

        # Encode classical input (data layer)
        self.q_device.ry(x[0], wires=0)
        self.q_device.ry(x[1], wires=1)
        self.q_device.cx(0, 1)

        # Apply weight parameters (variational layer)
        self.q_device.ry(w[0], wires=0)
        self.q_device.ry(w[1], wires=1)
        self.q_device.cx(0, 1)
        self.q_device.ry(w[2], wires=0)
        self.q_device.ry(w[3], wires=1)

        # Compute probabilities for |00⟩ and |01⟩
        probs = torch.abs(self.q_device.states[0])**2
        return probs[:2]  # return first two outcomes

__all__ = ["SamplerQNNGen"]
