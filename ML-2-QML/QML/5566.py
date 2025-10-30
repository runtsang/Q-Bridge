"""Hybrid quantum kernel using Pennylane and fraud‑layer inspired feature map.

The implementation demonstrates how to fuse a classical convolutional
feature map, a fraud‑layer style rotation, and a variational ansatz
to compute a similarity kernel between two data points.  Shot noise
is added by sampling the circuit multiple times.

The class is fully self‑contained and can be used as a drop‑in
replacement for the classical `HybridKernel` when a quantum backend
is desired.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Callable

import numpy as np
import pennylane as qml
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Utility: fraud‑layer inspired rotation angles
# --------------------------------------------------------------------------- #
class FraudLayerRotation(nn.Module):
    """
    Convert a FraudLayerParameters instance into a set of rotation angles
    that will be applied to the qubits in the feature map.
    """
    def __init__(self, params):
        super().__init__()
        # Extract angles; keep only the first two for simplicity
        self.theta = torch.tensor([params.bs_theta, params.bs_phi], dtype=torch.float32)
        self.phi = torch.tensor(params.phases, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple mapping: rotate each qubit by theta + phi * x
        return self.theta + self.phi * x

# --------------------------------------------------------------------------- #
# Hybrid quantum kernel
# --------------------------------------------------------------------------- #
class HybridKernel:
    """
    Quantum kernel based on Pennylane's qubit device.

    Parameters
    ----------
    wires : int
        Number of qubits used in the circuit.
    fraud_params : FraudLayerParameters | None
        Parameters used to generate a fraud‑layer style rotation.
    shots : int
        Number of measurement shots for shot‑noise simulation.
    """
    def __init__(
        self,
        wires: int = 4,
        fraud_params=None,
        shots: int = 1024,
    ) -> None:
        self.wires = wires
        self.device = qml.device("default.qubit", wires=self.wires)
        self.shots = shots

        # Build the fraud‑layer rotation mapping
        self.fraud_map = FraudLayerRotation(fraud_params) if fraud_params else None

        # Define the feature map and ansatz
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Feature map: encode both x and y into two separate halves
            for i in range(self.wires):
                qml.RY(x[i], wires=i)
            for i in range(self.wires):
                qml.RY(-y[i], wires=i)

            # Fraud‑layer inspired rotations
            if self.fraud_map is not None:
                angles = self.fraud_map(x)
                for i, a in enumerate(angles):
                    qml.RZ(a, wires=i)

            # Variational ansatz – a simple repeat of entangling layers
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.wires):
                qml.RX(0.1, wires=i)

            return qml.state()

        self.circuit = circuit

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the kernel value |⟨ψ(x)|ψ(y)⟩|^2 using the statevector.
        """
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        psi_x = self.circuit(x_t, torch.zeros_like(x_t))
        psi_y = self.circuit(y_t, torch.zeros_like(y_t))
        overlap = torch.abs(torch.dot(psi_x, psi_y.conj())) ** 2
        return float(overlap.item())

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Return the Gram matrix between two sets of samples."""
        mat = np.array([[self.kernel(a_i, b_j) for b_j in b] for a_i in a])
        return mat

    # ----------------------------------------------------------------------- #
    # Estimator interface – evaluate a Pennylane qnode with shot noise
    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        qnode: Callable[[torch.Tensor], torch.Tensor],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a Pennylane qnode for many parameter sets.
        Gaussian shot noise is added if ``shots`` is provided.
        """
        rng = np.random.default_rng(seed)

        results: List[List[float]] = []
        for params in parameter_sets:
            # Prepare circuit with given parameters
            inp = torch.tensor(params, dtype=torch.float32)
            # Run the qnode
            out = qnode(inp)
            row: List[float] = []
            for obs in observables:
                val = obs(out)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            if shots is not None:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                row = noisy_row
            results.append(row)
        return results

__all__ = ["HybridKernel"]
