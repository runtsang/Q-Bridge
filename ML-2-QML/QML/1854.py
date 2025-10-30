import pennylane as qml
import torch
import numpy as np
from typing import Iterable

class FCL:
    """Variational fully‑connected quantum layer.

    It builds a parameter‑dependent circuit on ``n_qubits`` with an initial
    Hadamard layer, followed by a trainable ``RY`` rotation on each qubit.
    The expectation value of ``PauliZ`` on the first qubit is returned.
    The circuit is constructed as a Pennylane QNode with a Torch interface,
    enabling automatic differentiation if desired.

    Args:
        n_qubits: Number of qubits in the circuit.
        device: Pennylane device string (e.g., ``default.qubit``).
        shots: Number of measurement shots.
    """
    def __init__(self, n_qubits: int = 1, device: str = "default.qubit", shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            # Prepare a uniform superposition
            qml.Hadamard(wires=range(n_qubits))
            # Apply parameterised rotations
            for i, p in enumerate(params):
                qml.RY(p, wires=i)
            # Measure the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit and return the expectation as a NumPy array."""
        params = torch.tensor(list(thetas), dtype=torch.float32, requires_grad=False)
        expectation = self.circuit(params)
        return expectation.detach().numpy()

__all__ = ["FCL"]
