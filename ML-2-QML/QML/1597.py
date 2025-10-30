"""Enhanced quantum self‑attention implementation.

Uses Pennylane to construct a variational circuit that mimics
multi‑head attention with configurable entanglement.  The public
factory ``SelfAttentionEnhanced`` returns an instance exposing
``run(backend, rotation_params, entangle_params, shots, batch)``.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

def SelfAttentionEnhanced(num_qubits: int = 8,
                          layers: int = 2,
                          entanglement: str = "circular") -> "QuantumSelfAttentionEnhanced":
    class QuantumSelfAttentionEnhanced:
        """Variational self‑attention circuit."""

        def __init__(self, num_qubits: int, layers: int, entanglement: str):
            self.num_qubits = num_qubits
            self.layers = layers
            self.entanglement = entanglement

            # Device for simulation
            self.dev = qml.device("default.qubit", wires=num_qubits)

            # Parameter shapes
            self.rotation_shape = (layers, num_qubits, 3)   # RX, RY, RZ
            self.entangle_shape = (layers, num_qubits - 1)  # Controlled‑X per qubit pair

            # Trainable parameters
            self.rotation_params = pnp.random.randn(*self.rotation_shape)
            self.entangle_params = pnp.random.randn(*self.entangle_shape)

            self._circuit = qml.QNode(self._build_circuit, self.dev, interface="numpy")

        def _build_circuit(self,
                           rotation_params: np.ndarray,
                           entangle_params: np.ndarray,
                           *inputs) -> np.ndarray:
            """Build the variational circuit for a single batch element."""
            # Encode the input tokens as angles on each qubit
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Apply layers
            for l in range(self.layers):
                # Rotations
                for q in range(self.num_qubits):
                    qml.RX(rotation_params[l, q, 0], wires=q)
                    qml.RY(rotation_params[l, q, 1], wires=q)
                    qml.RZ(rotation_params[l, q, 2], wires=q)

                # Entanglement
                if self.entanglement == "circular":
                    for q in range(self.num_qubits):
                        qml.CNOT(wires=[q, (q + 1) % self.num_qubits])
                elif self.entanglement == "linear":
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                else:
                    # Custom pattern via entangle_params
                    for q in range(self.num_qubits - 1):
                        if entangle_params[l, q] > 0:
                            qml.CNOT(wires=[q, q + 1])

            # Measurement in computational basis
            return qml.probs(wires=range(self.num_qubits))

        def run(self,
                backend: qml.device,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                shots: int = 1024,
                batch: np.ndarray = None) -> np.ndarray:
            """
            Executes the self‑attention circuit on the supplied backend.

            Parameters
            ----------
            backend : pennylane device
            rotation_params : shape (layers, num_qubits, 3)
            entangle_params : shape (layers, num_qubits-1)
            shots : number of measurement shots
            batch : array of shape (batch_size, num_qubits) of input angles
            Returns
            -------
            probs : array of shape (batch_size, 2**num_qubits)
            """
            if batch is None:
                raise ValueError("Batch of input angles must be provided.")

            # Vectorised execution
            probs = []
            for angles in batch:
                qml.set_options(device=backend, shots=shots)
                probs.append(self._circuit(rotation_params, entangle_params, *angles))
            return np.array(probs)

    return QuantumSelfAttentionEnhanced(num_qubits, layers, entanglement)

__all__ = ["SelfAttentionEnhanced"]
