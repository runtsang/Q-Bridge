"""Variational quantum circuit emulating a fully connected layer.

Uses Pennylane to construct a parameterized circuit with multiple qubits
and layers. The circuit outputs the expectation value of Pauli‑Z on the
last qubit, which serves as the layer’s activation.
"""
import pennylane as qml
import numpy as np
from typing import Iterable, Sequence

class FCL:
    """Variational quantum circuit acting as a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of alternating rotation‑and‑entanglement layers.
    device : str, optional
        Pennylane device name; defaults to ``default.qubit``.
    shots : int, optional
        Number of shots for expectation estimation; defaults to 1000.
    """
    def __init__(self, n_qubits: int, n_layers: int, device: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        # Total number of parameters: each layer has 3 rotations per qubit
        self.n_params = n_layers * n_qubits * 3

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor):
            # Encode parameters into Ry, Rz, Rx rotations per qubit per layer
            for l in range(n_layers):
                for q in range(n_qubits):
                    idx = l * n_qubits * 3 + q * 3
                    qml.Ry(params[idx], wires=q)
                    qml.Rz(params[idx + 1], wires=q)
                    qml.Rx(params[idx + 2], wires=q)
                # Entangle neighboring qubits with CNOTs
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Measurement expectation on last qubit
            return qml.expval(qml.PauliZ(wires=n_qubits - 1))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat sequence of parameters for all rotation gates.

        Returns
        -------
        np.ndarray
            Array containing the expectation value of Pauli‑Z on the last qubit.
        """
        if len(thetas)!= self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(thetas)}")
        import torch
        params = torch.tensor(thetas, dtype=torch.float32)
        expval = self._circuit(params)
        return np.array([expval.item()])

__all__ = ["FCL"]
