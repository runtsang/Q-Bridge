"""Quantum sampler using PennyLane with depth‑controlled variational circuit."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import pennylane as qml
import torch
from pennylane import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SamplerQNN"]


class SamplerQNN:
    """
    Variational quantum sampler.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    depth : int, default 2
        Number of entangling layers (each layer contains RY + CX + RY).
    device : str, default "default.qubit"
        PennyLane device name.
    seed : int, default 42
        Random seed for weight initialization.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        device: str = "default.qubit",
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device, wires=num_qubits, shots=1024)
        self.rng = np.random.default_rng(seed)

        # Parameter initialization
        self.input_params = qml.numpy.array(
            [0.0] * num_qubits, requires_grad=False, dtype=np.float64
        )
        self.weight_params = self.rng.uniform(
            low=-np.pi, high=np.pi, size=(depth, num_qubits)
        )

    def circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Build and execute the variational circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters of shape ``(num_qubits,)``.
        weights : np.ndarray
            Weight parameters of shape ``(depth, num_qubits)``.

        Returns
        -------
        np.ndarray
            Final statevector.
        """
        @qml.qnode(self.device, interface="torch", diff_method="parameter_shift")
        def run_circuit():
            # Input encoding (RY rotations)
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for d in range(self.depth):
                for q in range(self.num_qubits):
                    qml.RY(weights[d, q], wires=q)
                # Entangling pattern (cyclic CX)
                for q in range(self.num_qubits):
                    qml.CNOT(wires=(q, (q + 1) % self.num_qubits))

            return qml.probs(wires=range(self.num_qubits))

        return run_circuit()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning probability distribution over computational basis.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, num_qubits)`` representing input angles.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(batch, 2**num_qubits)``.
        """
        batch = inputs.shape[0]
        probs = torch.zeros(batch, 2 ** self.num_qubits)
        for i in range(batch):
            probs[i] = torch.tensor(
                self.circuit(inputs[i].detach().numpy(), self.weight_params), dtype=torch.float32
            )
        return probs

    def sample(self, probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the quantum probability distribution.

        Parameters
        ----------
        probs : torch.Tensor
            Probability tensor of shape ``(batch, 2**num_qubits)``.
        num_samples : int
            Number of samples per batch element.

        Returns
        -------
        torch.Tensor
            Integer indices of shape ``(batch, num_samples)``.
        """
        return torch.multinomial(probs, num_samples, replacement=True)

    def compute_loss(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int | None = None,
    ) -> torch.Tensor:
        """
        Negative log‑likelihood loss for quantum sampler.

        Parameters
        ----------
        probs : torch.Tensor
            Probabilities from :meth:`forward`.
        targets : torch.Tensor
            Long‑tensor of target basis indices.
        ignore_index : int or None
            Class index to ignore during loss computation.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        log_probs = torch.log(probs + 1e-12)
        loss = F.nll_loss(log_probs, targets, ignore_index=ignore_index)
        return loss
