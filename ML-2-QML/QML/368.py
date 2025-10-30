"""Quantum kernel construction with dynamic ansatz and batch support.

The implementation builds on TorchQuantum but can be swapped out for
other back‑ends (e.g. PennyLane or Qiskit) with minimal changes.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Programmable ansatz that encodes two classical vectors ``x`` and ``y``.

    Parameters
    ----------
    func_list : Iterable[Mapping[str, object]]
        List of dictionaries describing gates.  Each entry must contain:
        - ``input_idx``: list of feature indices used as parameters
        - ``func``: gate name (must be registered in ``tq.op_name_dict``)
        - ``wires``: list of qubit indices the gate acts upon
    """
    def __init__(self, func_list: Iterable[Mapping[str, object]]) -> None:
        super().__init__()
        self.func_list = list(func_list)

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """Apply the ansatz to the device.

        The device is reset for every batch.  ``x`` and ``y`` must
        have the same batch dimension and feature length.
        """
        batch_size = x.shape[0]
        q_device.reset_states(batch_size)

        # Encode x
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        # Encode y with reversed sign
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits used in the device.
    func_list : Iterable[Mapping[str, object]], optional
        Custom ansatz.  If ``None`` a default 4‑qubit ``ry`` encoder is used.
    """
    def __init__(
        self,
        n_wires: int = 4,
        func_list: Iterable[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        default_ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

        self.ansatz = KernalAnsatz(func_list or default_ansatz)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return the absolute overlap |<x|y>| for each batch element.

        The method accepts 2‑D tensors of shape ``(n, d)``.  For
        compatibility with the original API it also accepts 1‑D tensors.
        """
        x = x if x.ndim == 2 else x.unsqueeze(0)
        y = y if y.ndim == 2 else y.unsqueeze(0)
        self.ansatz(self.q_device, x, y)
        # The first element of the state vector is the amplitude of |0...0>
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(
    a: Union[Sequence[torch.Tensor], torch.Tensor],
    b: Union[Sequence[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """Batch‑wise Gram matrix evaluation using the quantum kernel.

    Parameters
    ----------
    a, b : iterable of tensors or 2‑D tensors
        Input datasets.  Each element is a feature vector.
    """
    if isinstance(a, (list, tuple)):
        a = torch.stack(a)
    if isinstance(b, (list, tuple)):
        b = torch.stack(b)

    kernel = Kernel()
    return kernel(a, b).detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
