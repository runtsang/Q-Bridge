"""Quantum implementation of a hybrid kernel method using TorchQuantum.

This class mirrors the classical interface but evaluates the kernel
by encoding data into a quantum circuit.  The circuit consists of a
classical linear feature map that produces parameters for a
parameterised ansatz.  The default ansatz is a hardware‑efficient
layer of Ry rotations followed by a layer of CNOT entangling gates.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel with classical pre‑processor.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits in the ansatz.
    ansatz : list[dict], optional
        List describing the variational circuit.  Each dict must
        contain ``"func"`` (gate name), ``"wires"`` (list of qubit
        indices) and ``"input_idx"`` (index into the feature map).
    feature_dim : int, default=1
        Dimensionality of the classical feature map.
    device : str or torch.device, default='cpu'
        Quantum device backend.
    """

    def __init__(self,
                 n_wires: int = 4,
                 ansatz: list[dict] | None = None,
                 feature_dim: int = 1,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires, device=device)
        # Classical pre‑processor that maps input to rotation angles
        self.feature_map = nn.Linear(feature_dim, n_wires, bias=False).to(device)

        if ansatz is None:
            # Default hardware‑efficient ansatz: Ry on each qubit,
            # followed by a CNOT chain for entanglement.
            self.ansatz = [
                {"func": "ry", "wires": [i], "input_idx": i} for i in range(n_wires)
            ] + [
                {"func": "cx", "wires": [i, i + 1], "input_idx": None} for i in range(n_wires - 1)
            ]
        else:
            self.ansatz = ansatz

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate quantum kernel between two 1‑D tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape (n,) or (batch, n).

        Returns
        -------
        torch.Tensor
            Scalar kernel value or batch of values.
        """
        x = x.to(self.q_device.device).float().view(-1, 1)
        y = y.to(self.q_device.device).float().view(-1, 1)

        # Reset device
        self.q_device.reset_states(1)

        # Encode x
        params_x = self.feature_map(x)  # shape (1, n_wires)
        for gate in self.ansatz:
            if gate["func"] == "cx":
                # Entangling gate has no parameters
                func_name_dict[gate["func"]](self.q_device,
                                             wires=gate["wires"])
            else:
                idx = gate["input_idx"]
                param = params_x[:, idx].unsqueeze(-1)
                func_name_dict[gate["func"]](self.q_device,
                                             wires=gate["wires"],
                                             params=param)

        # Encode y with negative sign (inverse of the circuit)
        params_y = -self.feature_map(y)
        for gate in reversed(self.ansatz):
            if gate["func"] == "cx":
                func_name_dict[gate["func"]](self.q_device,
                                             wires=gate["wires"])
            else:
                idx = gate["input_idx"]
                param = params_y[:, idx].unsqueeze(-1)
                func_name_dict[gate["func"]](self.q_device,
                                             wires=gate["wires"],
                                             params=param)

        # Overlap of the first basis state after circuit
        return torch.abs(self.q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        km = QuantumKernelMethod()
        return np.array([[km(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod"]
