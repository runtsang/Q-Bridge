"""Quantum kernel construction using Pennylane with automatic differentiation support.

The module defines a class QuantumKernelMethod that evaluates a quantum kernel
using a fixed ansatz. The ansatz encodes two classical vectors via a sequence
of rotations and measures the overlap. The kernel can be differentiated
with respect to the circuit parameters if needed.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Sequence, Iterable

class QuantumAnsatz(qml.QuantumNode):
    """Encodes two feature vectors into a quantum state and returns the overlap."""
    def __init__(self, func_list):
        super().__init__(self._ansatz, device="default.qubit", wires=4)
        self.func_list = func_list

    def _ansatz(self, x: np.ndarray, y: np.ndarray):
        dev = self.device
        dev.reset()
        for info in self.func_list:
            params = x[info["input_idx"]] if qml.operation.operation_from_name(info["func"]).num_params else None
            getattr(qml, info["func"])(wires=info["wires"], parameters=params)
        for info in reversed(self.func_list):
            params = -y[info["input_idx"]] if qml.operation.operation_from_name(info["func"]).num_params else None
            getattr(qml, info["func"])(wires=info["wires"], parameters=params)
        return qml.expval(qml.PauliX(0))

class QuantumKernel(QuantumAnsatz):
    """Quantum kernel that uses a fixed 4‑qubit ansatz."""
    def __init__(self):
        func_list = [
            {"input_idx": 0, "func": "RY", "wires": [0]},
            {"input_idx": 1, "func": "RY", "wires": [1]},
            {"input_idx": 2, "func": "RY", "wires": [2]},
            {"input_idx": 3, "func": "RY", "wires": [3]},
        ]
        super().__init__(func_list)

class QuantumKernelMethod:
    """Interface that exposes the quantum kernel and its Gram matrix."""
    def __init__(self):
        self.kernel = QuantumKernel()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return kernel value for two classical vectors."""
        return float(self.kernel(x, y))

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Compute the Gram matrix for two lists of vectors."""
        K = np.array([[self.evaluate(x, y) for y in b] for x in a])
        return K

    def kernel_matrix_multibatch(self, datasets: Iterable[Sequence[np.ndarray]]) -> np.ndarray:
        """Compute a block‑wise Gram matrix for a list of datasets."""
        all_tensors = [np.vstack(ds) for ds in datasets]
        combined = np.concatenate(all_tensors, axis=0)
        combined_list = [combined[i] for i in range(combined.shape[0])]
        return self.kernel_matrix(combined_list, combined_list)

__all__ = ["QuantumKernelMethod"]
