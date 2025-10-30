"""
Hybrid quantum kernel module.

Features
--------
* Parameterised ansatz with depth control and optional entanglement.
* Supports multiple device backends (CPU, GPU) via TorchQuantum.
* Flexible measurement: inner‑product overlap or arbitrary POVM.

The public API mirrors the classical version for easy comparison.
"""

from __future__ import annotations

from typing import Sequence, List, Dict, Any
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import QuantumDevice


class HybridKernel(tq.QuantumModule):
    """
    Quantum kernel evaluated with a depth‑controlled, entangling ansatz.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits.
    depth : int, default 2
        Number of circuit layers.
    ansatz_type : str, default 'ry'
        Rotation gate used for data encoding ('ry', 'rx', 'rz', 'crx', etc.).
    entangle : bool, default True
        Whether to insert CNOT entangling gates between adjacent qubits.
    device_type : str, default 'cpu'
        Quantum device backend ('cpu', 'gpu', 'qiskit', etc.).
    """

    def __init__(
        self,
        *,
        n_wires: int = 4,
        depth: int = 2,
        ansatz_type: str = "ry",
        entangle: bool = True,
        device_type: str = "cpu",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.ansatz_type = ansatz_type
        self.entangle = entangle
        self.q_device = QuantumDevice(n_wires=n_wires, device=device_type)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> List[Dict[str, Any]]:
        """
        Construct a list of dictionaries describing the gates in the ansatz.
        Each entry contains ``func`` (gate name), ``wires`` (target qubits),
        and ``input_idx`` (indices of the input data features).
        """
        func = self.ansatz_type
        ansatz: List[Dict[str, Any]] = []

        # Data‑encoding layer
        for i in range(self.depth):
            for w in range(self.n_wires):
                ansatz.append(
                    {
                        "func": func,
                        "wires": [w],
                        "input_idx": [w % self.n_wires],
                    }
                )
            if self.entangle:
                # Entangle adjacent qubits
                for w in range(self.n_wires - 1):
                    ansatz.append(
                        {
                            "func": "cnot",
                            "wires": [w, w + 1],
                            "input_idx": [],
                        }
                    )
        return ansatz

    @tq.static_support
    def forward(self, q_device: QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode data vectors x and y onto the quantum device and apply the ansatz.

        Parameters
        ----------
        q_device : QuantumDevice
            The quantum device to run on.
        x, y : torch.Tensor
            Input vectors of shape ``(N, D)`` and ``(M, D)``.
        """
        # Reset and encode x
        q_device.reset_states(x.shape[0])
        for gate in self.ansatz:
            params = (
                x[:, gate["input_idx"]]
                if gate["input_idx"]
                else None
            )
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

        # Apply inverse encoding for y (negative rotation)
        for gate in reversed(self.ansatz):
            params = (
                -y[:, gate["input_idx"]]
                if gate["input_idx"]
                else None
            )
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

    def gram(self, X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the Gram matrix between datasets X and Y.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(N, D)``.
        Y : torch.Tensor, optional
            Shape ``(M, D)``.  If ``None``, Y = X.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(N, M)``.
        """
        Y = Y if Y is not None else X
        N, M = X.shape[0], Y.shape[0]
        gram = torch.empty((N, M), device=X.device)

        for i in range(N):
            for j in range(M):
                self.forward(self.q_device, X[i : i + 1], Y[j : j + 1])
                # Overlap of the first amplitude (|0...0> component)
                gram[i, j] = torch.abs(self.q_device.states.view(-1)[0])

        return gram

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Convenience wrapper returning a NumPy array.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(len(a), len(b))``.
        """
        X = torch.stack(a)
        Y = torch.stack(b)
        return self.gram(X, Y).cpu().numpy()


__all__ = ["HybridKernel", "kernel_matrix"]
