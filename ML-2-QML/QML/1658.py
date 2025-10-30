"""HybridConvLayer: quantum filter implemented with PennyLane.

This module implements a variational quantum circuit that acts as a
convolutional filter.  It maps a 2×2 patch of classical data to a
scalar output in [0, 1] by measuring the expectation value of a
product‑state observable.  The circuit is fully parameterised and
supports gradient computation via the parameter‑shift rule, enabling
end‑to‑end training of a hybrid model.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Tuple

class HybridConvLayer:
    """Quantum convolutional filter using PennyLane."""
    def __init__(
        self,
        kernel_size: int = 2,
        device_name: str = "default.qubit",
        shots: int = 1000,
        threshold: float = 0.5,
        init: str = "random",
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        self.dev = qml.device(device_name, wires=self.n_qubits, shots=shots)

        if init == "random":
            self.params = pnp.random.uniform(0, 2 * np.pi, self.n_qubits)
        else:
            self.params = pnp.zeros(self.n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x: np.ndarray, params: np.ndarray):
            for i, val in enumerate(x.flatten()):
                if val > self.threshold:
                    qml.RX(np.pi, wires=i)
                else:
                    qml.RX(0.0, wires=i)

            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
                if i < self.n_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """Run the quantum filter on a 2D patch."""
        out = self.circuit(data, self.params)
        return float((out + 1) / 2)

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """Compute gradient of the output w.r.t. the variational parameters."""
        return self.circuit.grad(data, self.params)

__all__ = ["HybridConvLayer"]
