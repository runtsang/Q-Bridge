"""Quantum convolution filter implemented with Pennylane.

The Conv function returns a callable object that runs a variational
circuit on a 2‑D patch of data.  The circuit encodes the data into
RY rotations, applies a set of learnable RX rotations, entangles the
qubits with a chain of CNOTs, and measures the probability of
observing |1> on each qubit.  The final value is the mean probability
across all qubits.  A threshold can be set to binarise the input
before encoding, mirroring the behaviour of the classical filter.
"""

import pennylane as qml
import numpy as np
from typing import Optional, Dict

__all__ = ["Conv"]


def Conv(kernel_size: int = 3,
         threshold: float = 0.5,
         shots: int = 1024,
         dev_type: str = "default.qubit",
         dev_kwargs: Optional[Dict] = None) -> object:
    """Create a quantum convolution filter.

    Parameters
    ----------
    kernel_size : int
        Size of the square patch.  The number of qubits is ``kernel_size**2``.
    threshold : float
        Threshold applied to the classical data before encoding.
    shots : int
        Number of shots for simulator execution.
    dev_type : str
        Pennylane device type.
    dev_kwargs : dict | None
        Extra keyword arguments passed to pennylane.device.

    Returns
    -------
    object
        An object with a ``run`` method that accepts a 2‑D numpy array of shape
        ``(kernel_size, kernel_size)`` and returns a float.
    """
    if dev_kwargs is None:
        dev_kwargs = {}

    n_qubits = kernel_size ** 2
    dev = qml.device(dev_type, wires=n_qubits, shots=shots, **dev_kwargs)

    # Initialize parameters for the variational circuit
    init_params = np.random.uniform(0, 2 * np.pi, size=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params, data):
        """Variational circuit that encodes input data and applies learnable rotations."""
        # Encode data: RY rotations proportional to (value - threshold)
        for i, val in enumerate(data.flat):
            qml.RY((val - threshold) * np.pi, wires=i)
        # Learnable RX rotations
        for i, p in enumerate(params):
            qml.RX(p, wires=i)
        # Simple linear chain of CNOTs to entangle qubits
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Return probability distribution for each qubit
        return [qml.probs(wires=i) for i in range(n_qubits)]

    class QuanvCircuit:
        def __init__(self):
            self.params = init_params
            self.dev = dev

        def run(self, data: np.ndarray) -> float:
            """Run the circuit on a single patch and return the average |1> probability."""
            probs_per_qubit = circuit(self.params, data)
            # probs_per_qubit is a list of length n_qubits; each element is a 2‑element array
            # with probabilities of |0> and |1>.  We take the probability of |1>.
            ones = np.array([p[1] for p in probs_per_qubit])
            return float(np.mean(ones))

    return QuanvCircuit()
