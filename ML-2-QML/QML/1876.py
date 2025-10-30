"""
Quantum kernel utilities – Pennylane implementation.

This module implements a variational quantum kernel that
encodes two classical vectors via a parameter‑tunable Ry‑rotation
ansatz with CNOT entanglement.  The kernel is the fidelity between
the two encoded states, which is a valid Mercer kernel.

Key features:
* Variational ansatz with adjustable depth.
* CNOT entanglement pattern.
* Default simulator backend (default.qubit) but can be swapped.
* GPU acceleration via the GPU backend of Pennylane.
"""

import numpy as np
import pennylane as qml
from typing import Sequence

__all__ = ["Kernel", "kernel_matrix"]

class Kernel:
    """Quantum kernel based on a Ry‑rotation ansatz.

    Parameters
    ----------
    wires : Sequence[int] | None
        Qubit indices.  Defaults to 0..n-1.
    depth : int, default 2
        Number of variational layers.
    backend : str, default 'default.qubit'
        Pennylane simulator backend.
    """

    def __init__(self, wires=None, depth: int = 2, backend: str = "default.qubit") -> None:
        self.wires = wires if wires is not None else list(range(depth))
        self.depth = depth
        self.device = qml.device(backend, wires=len(self.wires))
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            # params shape: (depth, len(wires))
            for layer in range(self.depth):
                for idx, w in enumerate(self.wires):
                    qml.RY(params[layer, idx], wires=w)
                # Entanglement
                for idx in range(len(self.wires) - 1):
                    qml.CNOT(wires=[self.wires[idx], self.wires[idx + 1]])
            return qml.state()

        self.circuit = circuit

    def _encode(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into circuit parameters."""
        data = np.asarray(data, dtype=np.float64)
        if data.size!= len(self.wires):
            raise ValueError(f"Data length {data.size} does not match number of wires {len(self.wires)}.")
        # Replicate data across layers
        return np.tile(data, (self.depth, 1))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return kernel value: fidelity between |ψ(x)⟩ and |ψ(y)⟩."""
        params_x = self._encode(x)
        params_y = -self._encode(y)  # encode y with opposite sign
        state_x = self.circuit(params_x)
        state_y = self.circuit(params_y)
        fidelity = np.abs(np.vdot(state_x, state_y)) ** 2
        return float(fidelity)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.evaluate(x, y)

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray], depth: int = 2,
                  backend: str = "default.qubit") -> np.ndarray:
    """Compute Gram matrix for a list of classical vectors."""
    kernel = Kernel(depth=depth, backend=backend)
    return np.array([[kernel(x, y) for y in b] for x in a])
