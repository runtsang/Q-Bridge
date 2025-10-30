"""Quantum sampler implemented with PennyLane.

The circuit receives a 2‑dimensional input vector and a 4‑dimensional set of rotation angles.
It applies a sequence of RY and CX gates, mirroring the original SamplerQNN design, and
measures the probability of the first qubit being in state |0⟩.  The returned probabilities
are compatible with the classical network's output shape.
"""

from __future__ import annotations

import pennylane as qml
import torch
import numpy as np


class QuantumSampler:
    """Variational quantum sampler using PennyLane."""

    def __init__(self, dev_name: str = "default.qubit") -> None:
        self.dev = qml.device(dev_name, wires=2)

    @qml.qnode
    def _circuit(self, input_params: np.ndarray, weight_params: np.ndarray) -> np.ndarray:
        qml.RY(input_params[0], wires=0)
        qml.RY(input_params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weight_params[0], wires=0)
        qml.RY(weight_params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weight_params[2], wires=0)
        qml.RY(weight_params[3], wires=1)
        return qml.probs([0])

    def __call__(self, input_params: torch.Tensor, weight_params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_params : torch.Tensor
            Shape (batch, 2) – input vector for the quantum circuit.
        weight_params : torch.Tensor
            Shape (batch, 4) – rotation angles for the variational circuit.

        Returns
        -------
        torch.Tensor
            Shape (batch, 2) – probability of qubit 0 being |0⟩ or |1⟩.
        """
        batch_size = input_params.shape[0]
        probs_list = []
        for i in range(batch_size):
            inp = input_params[i].detach().cpu().numpy()
            wgt = weight_params[i].detach().cpu().numpy()
            probs = self._circuit(inp, wgt)
            probs_list.append(probs)
        probs_arr = np.stack(probs_list, axis=0)
        return torch.from_numpy(probs_arr).float()


__all__ = ["QuantumSampler"]
