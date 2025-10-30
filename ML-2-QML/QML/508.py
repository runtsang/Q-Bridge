"""Quantum kernel with a trainable variational ansatz using TorchQuantum."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import List, Dict, Any, Sequence, Union, Optional

class QuantumKernelEnhanced(tq.QuantumModule):
    """
    Variational quantum kernel that supports trainable parameters.
    """
    def __init__(self,
                 n_wires: int = 4,
                 ansatz: Optional[List[Dict[str, Any]]] = None,
                 trainable: bool = True) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.trainable = trainable

        if ansatz is None:
            ansatz = [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]

        self.ansatz = ansatz

        if self.trainable:
            self.param_list = []
            for gate in ansatz:
                if tq.op_name_dict[gate["func"]].num_params:
                    self.param_list.append(
                        torch.nn.Parameter(torch.zeros(len(gate["input_idx"])))
                    )
            self.params = torch.nn.ParameterList(self.param_list)

    def _apply_gate(self, x: torch.Tensor, y: torch.Tensor,
                    params: Optional[torch.Tensor] = None) -> None:
        """
        Encode data and unencode data with the ansatz.
        """
        self.q_device.reset_states(x.shape[0])
        for idx, gate in enumerate(self.ansatz):
            func = func_name_dict[gate["func"]]
            if func.num_params:
                gate_params = x[:, gate["input_idx"]] if params is None else params[idx]
                func(self.q_device, wires=gate["wires"], params=gate_params)
        for idx, gate in reversed(list(enumerate(self.ansatz))):
            func = func_name_dict[gate["func"]]
            if func.num_params:
                gate_params = -y[:, gate["input_idx"]] if params is None else -params[idx]
                func(self.q_device, wires=gate["wires"], params=gate_params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the overlap amplitude |⟨ψ(x)|ψ(y)⟩|² for batches of samples.
        """
        x_exp = x.unsqueeze(1)  # (n, 1, d)
        y_exp = y.unsqueeze(0)  # (1, m, d)
        n, m, _ = x_exp.shape[0], y_exp.shape[1], x_exp.shape[2]

        if self.trainable:
            params = [p.repeat(n * m, 1) for p in self.params]
        else:
            params = None

        K = torch.empty(n, m, dtype=torch.float32, device=x.device)
        for i in range(n):
            for j in range(m):
                self._apply_gate(x_exp[i], y_exp[j], params)
                # Overlap amplitude squared
                K[i, j] = torch.abs(self.q_device.states.view(-1)[0]) ** 2
        return K

def kernel_matrix(a: Sequence[Union[torch.Tensor, np.ndarray]],
                  b: Sequence[Union[torch.Tensor, np.ndarray]]) -> np.ndarray:
    """
    Compute Gram matrix of the quantum kernel for two datasets.
    """
    if not isinstance(a[0], torch.Tensor):
        a = [torch.from_numpy(np.asarray(x)) for x in a]
    if not isinstance(b[0], torch.Tensor):
        b = [torch.from_numpy(np.asarray(x)) for x in b]
    a = torch.stack(a)
    b = torch.stack(b)
    kernel = QuantumKernelEnhanced()
    K = kernel(a, b)
    return K.detach().cpu().numpy()

# Backwards‑compatibility aliases
KernalAnsatz = QuantumKernelEnhanced
Kernel = QuantumKernelEnhanced

__all__ = ["QuantumKernelEnhanced", "kernel_matrix", "KernalAnsatz", "Kernel"]
