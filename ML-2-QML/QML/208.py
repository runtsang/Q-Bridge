import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Tuple


class SelfAttentionFusion:
    """
    Quantum self‑attention block implemented with Pennylane.
    The circuit implements a variational self‑attention style unit:
    - Inputs are angle‑encoded into qubits.
    - rotation_params control single‑qubit rotations.
    - entangle_params control two‑qubit controlled‑RZ gates between neighbours.
    The measurement of Pauli‑Z expectation values is interpreted as the
    attention output.
    """

    def __init__(
        self,
        n_qubits: int,
        embed_dim: int,
        device: str = "default.qubit",
        shots: int = 1024,
    ):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit (must be >= embed_dim).
        embed_dim : int
            Dimensionality of the output vector.
        device : str, optional
            Pennylane device name.
        shots : int, optional
            Number of shots for sampling.
        """
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.shots = shots

        if n_qubits < embed_dim:
            raise ValueError("n_qubits must be >= embed_dim for output extraction.")

        self.dev = qml.device(device, wires=n_qubits, shots=shots)

    def _circuit(self, rotation_params, entangle_params, inputs):
        """Variational circuit implementing a self‑attention style block."""
        # Angle encoding of the input data
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # Rotation parameters (single‑qubit rotations)
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)

        # Entanglement parameters (controlled‑RZ gates)
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        # Measure expectation values of Z on first embed_dim qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.

        Parameters
        ----------
        backend : str or pennylane.Device
            Quantum back‑end (e.g., 'default.qubit', 'qiskit.ibmq', etc.).
        rotation_params : np.ndarray
            Parameters for single‑qubit rotations (size 3*n_qubits).
        entangle_params : np.ndarray
            Parameters for two‑qubit entanglement gates (size n_qubits-1).
        inputs : np.ndarray
            Input vector of length n_qubits (or padded to n_qubits).

        Returns
        -------
        np.ndarray
            Quantum‑derived attention output of shape (embed_dim,).
        """
        if not isinstance(backend, qml.Device):
            self.dev = qml.device(backend, wires=self.n_qubits, shots=self.shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            return self._circuit(rotation_params, entangle_params, inputs)

        raw_output = circuit()
        # Convert raw expectation values to float array
        return np.array(raw_output)

    def validate(self, classical_outputs: np.ndarray, tolerance: float = 0.1) -> bool:
        """
        Validate that quantum outputs are close to the provided classical outputs.

        Parameters
        ----------
        classical_outputs : np.ndarray
            Classical attention outputs for comparison.
        tolerance : float, optional
            Acceptable difference threshold.

        Returns
        -------
        bool
            True if the maximum absolute difference is below the tolerance.
        """
        quantum_outputs = self.run(
            backend=self.dev,
            rotation_params=np.zeros(3 * self.n_qubits),
            entangle_params=np.zeros(self.n_qubits - 1),
            inputs=np.zeros(self.n_qubits),
        )
        diff = np.abs(quantum_outputs - classical_outputs)
        return np.all(diff < tolerance)
