"""Variational quantum kernel for hybrid kernel learning.

The module implements a depth‑controlled, trainable quantum circuit that
encodes two classical feature vectors via parameterised rotations and
returns the overlap of the resulting states.  It is compatible with
TorchQuantum and can be used as the `quantum_kernel_cls` argument in
the :class:`HybridKernelRegressor` defined above.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelVar(tq.QuantumModule):
    """
    Trainable quantum kernel with a shallow variational ansatz.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the device.
    depth : int
        Number of repeating layers in the circuit.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Create a list of parameter names for each layer
        self.params: list[torch.nn.Parameter] = []
        for layer in range(self.depth):
            # For each qubit we will use a ry gate
            for qubit in range(self.n_wires):
                p = nn.Parameter(torch.rand(1))
                self.params.append(p)

        # Register parameters in the module
        self.register_parameter("params", nn.ParameterList(self.params))

        # Build a static list of operations for forward
        self.circuit_ops = []
        for layer in range(self.depth):
            for qubit in range(self.n_wires):
                self.circuit_ops.append({
                    "func": "ry",
                    "wires": [qubit],
                    "param_index": layer * self.n_wires + qubit
                })

    # --------------------------------------------------------------------- #
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode x and y on the same device and compute the overlap.

        Parameters
        ----------
        q_device : torchquantum.QuantumDevice
        x : torch.Tensor, shape (batch, n_features)
        y : torch.Tensor, shape (batch, n_features)
        """
        batch = x.shape[0]
        q_device.reset_states(batch)

        # Encode x
        for op in self.circuit_ops:
            p_idx = op["param_index"]
            param = x[:, p_idx % self.n_wires] if op["func"] in func_name_dict else None
            func_name_dict[op["func"]](q_device, wires=op["wires"], params=param)

        # Un-encode y with negative parameters to compute overlap
        for op in reversed(self.circuit_ops):
            p_idx = op["param_index"]
            param = -y[:, p_idx % self.n_wires] if op["func"] in func_name_dict else None
            func_name_dict[op["func"]](q_device, wires=op["wires"], params=param)

    # --------------------------------------------------------------------- #
    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return a scalar kernel value for a pair of inputs.

        Parameters
        ----------
        x, y : torch.Tensor, shape (batch, n_features)

        Returns
        -------
        torch.Tensor of shape (batch, 1)
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(-1)

    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix between two datasets using the variational kernel.

        Parameters
        ----------
        a, b : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        torch.Tensor of shape (n_samples, n_samples)
        """
        n, _ = a.shape
        m, _ = b.shape
        a_exp = a.unsqueeze(1).expand(n, m, -1)
        b_exp = b.unsqueeze(0).expand(n, m, -1)
        return self.kernel_value(a_exp.reshape(-1, a.shape[1]),
                                 b_exp.reshape(-1, b.shape[1])).reshape(n, m)

__all__ = ["QuantumKernelVar"]
