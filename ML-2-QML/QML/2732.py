"""Quantum kernel with optional photonic‑style feature extraction.

The quantum kernel is implemented using TorchQuantum.  The
architecture is a two‑qubit ansatz that encodes data via RY gates
followed by a reverse encoding.  In addition, a photonic‑style
pre‑processor can be attached using the same `FraudLayerParameters`
but expressed in terms of generic parametric gates that emulate
beam‑splitters, squeezers, displacements and Kerr interactions.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Import fraud‑detection utilities from the same package
from.FraudDetection import FraudLayerParameters

__all__ = ["QuantumKernelMethod"]


class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum RBF‑style kernel with optional photonic‑style feature extractor.
    """

    def __init__(
        self,
        n_wires: int = 4,
        use_photonic: bool = False,
        photonic_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_wires : int
            Number of qubits in the device.
        use_photonic : bool
            If True, a photonic‑style pre‑processor is added.
        photonic_params : Iterable[FraudLayerParameters], optional
            Parameters for the photonic pre‑processor.  Ignored if
            ``use_photonic`` is False.
        """
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()
        self.use_photonic = use_photonic
        self.photonic_extractor: Optional[tq.QuantumModule] = None
        if self.use_photonic and photonic_params is not None:
            self.photonic_extractor = self._build_photonic_extractor(photonic_params)

    def _build_ansatz(self) -> tq.QuantumModule:
        """Standard data‑encoding ansatz."""
        return tq.QuantumModule(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def _build_photonic_extractor(
        self, params: Iterable[FraudLayerParameters]
    ) -> tq.QuantumModule:
        """Photonic‑style pre‑processor built from FraudLayerParameters."""
        func_list = []
        for p in params:
            # Map photonic operations to generic parametric gates
            func_list.append({"input_idx": [0], "func": "rx", "wires": [0]})   # BS → RX
            func_list.append({"input_idx": [1], "func": "rz", "wires": [1]})   # BS → RZ
            func_list.append({"input_idx": [0], "func": "ry", "wires": [0]})   # Squeeze → RY
            func_list.append({"input_idx": [1], "func": "ry", "wires": [1]})   # Squeeze → RY
        return tq.QuantumModule(func_list)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        if self.photonic_extractor is not None:
            self.photonic_extractor(self.q_device, x, y)

        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the Gram matrix for two collections of samples."""
        kernel = self
        return np.array([[kernel(x, y).item() for y in b] for x in a])
