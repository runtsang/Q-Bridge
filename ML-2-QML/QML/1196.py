"""Quantum kernel construction using Pennylane.

The class implements a variational quantum kernel that can be trained
via automatic differentiation.  The default ansatz is a single‑layer
parameter‑shared rotation on each qubit, but the user can provide a
custom circuit function.

The kernel is defined as the squared absolute overlap between two
encoded states:
    K(x, y) = |⟨ψ(x)|ψ(y)⟩|²
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pennylane as qml
import torch


class QuantumKernelMethod:
    """Variational quantum kernel with trainable ansatz.

    Parameters
    ----------
    wires : int, optional
        Number of qubits used in the circuit.
    device : pennylane.Device, optional
        Backend device.  If ``None`` a default ``default.qubit`` device
        is created.
    ansatz : Callable[[qml.Device, torch.Tensor, int], None], optional
        User‑supplied circuit that encodes a single sample.  The function
        receives the device, the sample tensor, and the index of the
        sample in a batch.  If ``None`` a simple rotation‑only
        ansatz is used.
    """

    def __init__(
        self,
        wires: int = 4,
        device: qml.Device | None = None,
        ansatz: Callable[[qml.Device, torch.Tensor, int], None] | None = None,
    ) -> None:
        self.wires = wires
        self.device = device or qml.device("default.qubit", wires=self.wires)
        self.ansatz = ansatz or self._default_ansatz
        # Parameters of the ansatz are stored as a torch.Parameter
        # and updated via back‑propagation.
        self.params = torch.nn.Parameter(torch.randn(self.wires, dtype=torch.float32))

    def _default_ansatz(self, dev: qml.Device, sample: torch.Tensor, idx: int) -> None:
        """Simple rotation‑only encoding."""
        for w in range(self.wires):
            qml.RY(sample[w] * self.params[w], wires=w)

    def _statevector(self, sample: torch.Tensor) -> torch.Tensor:
        """Return the statevector for a given sample."""
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit() -> torch.Tensor:
            self.ansatz(self.device, sample, 0)
            return qml.state()

        return circuit()

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two samples."""
        psi_x = self._statevector(x)
        psi_y = self._statevector(y)
        # Squared absolute overlap
        return torch.abs(torch.dot(psi_x.conj(), psi_y)) ** 2

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sets of samples."""
        K = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                K[i, j] = self.kernel(x, y).item()
        return K
