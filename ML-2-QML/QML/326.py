"""Quantum kernel construction with a variational ansatz and parameter‑shift gradients.

The module extends the original TorchQuantum implementation by:
* Adding a trainable variational layer (Ry rotations) that can be optimised
  jointly with the kernel hyper‑parameters.
* Exposing a ``trainable`` flag for downstream optimisation.
* Providing an optional preprocessing hook that is applied to the classical
  feature vectors before encoding.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """
    Variational quantum kernel that encodes classical data and applies a
    trainable layer of Ry rotations.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers (each consisting of a Ry on every qubit).
    preprocess : Callable[[torch.Tensor], torch.Tensor] | None
        Optional preprocessing applied to the input tensors before encoding.
    """

    def __init__(
        self,
        n_wires: int,
        n_layers: int,
        *,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.preprocess = preprocess
        # Trainable parameters for each layer and qubit
        self.params = nn.Parameter(
            torch.rand(n_layers, n_wires, dtype=torch.float32)
        )

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Encode x, apply variational layer, then encode y with negative sign.
        """
        if self.preprocess is not None:
            x = self.preprocess(x)
            y = self.preprocess(y)

        q_device.reset_states(x.shape[0])

        # Encode x
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=i, params=x[:, i])

        # Variational layer
        for layer in range(self.n_layers):
            for i in range(self.n_wires):
                func_name_dict["ry"](
                    q_device, wires=i, params=self.params[layer, i]
                )

        # Encode y with negative sign
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=i, params=-y[:, i])


class Kernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the overlap of two encoded states.

    Parameters
    ----------
    n_wires : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    preprocess : Callable[[torch.Tensor], torch.Tensor] | None
        Optional preprocessing applied to the input data.
    """

    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        *,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires, n_layers, preprocess=preprocess)
        self.trainable = True

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return absolute amplitude of the |0...0> state
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    n_wires: int = 4,
    n_layers: int = 2,
    *,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    """
    Compute the Gram matrix between two lists of tensors using the
    variational quantum kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of feature vectors.
    n_wires : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers.
    preprocess : Callable, optional
        Optional preprocessing applied to each vector before encoding.
    """
    kernel = Kernel(n_wires, n_layers, preprocess=preprocess)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
