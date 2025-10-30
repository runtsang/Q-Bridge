"""Hybrid quantum filter that maps 2×2 image patches to measurement probabilities.

The circuit uses parameterized RY gates driven by data, a deterministic
entangling layer, and measures Pauli‑Z expectation values to approximate
the probability of observing |1> on each qubit. The interface mirrors the
classical Conv() function for seamless integration.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

def Conv() -> "HybridQuanConvCircuit":
    class HybridQuanConvCircuit:
        def __init__(self, patch_size: int = 2, shots: int = 100, threshold: float = 0.5) -> None:
            self.patch_size = patch_size
            self.shots = shots
            self.threshold = threshold
            self.n_qubits = patch_size * patch_size
            self.device = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
            self.qnode = qml.QNode(self._circuit, self.device)

        def _circuit(self, params: np.ndarray) -> list[float]:
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # deterministic entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        def run(self, data: np.ndarray) -> float:
            """Run the quantum filter on a 2×2 patch and return average |1> probability."""
            flat = data.flatten()
            expvals = np.array(self.qnode(flat))
            probs = (1 - expvals) / 2  # convert Z expectation to |1> probability
            return probs.mean()

    return HybridQuanConvCircuit()
