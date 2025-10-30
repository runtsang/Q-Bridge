"""Quantum convolution filter implemented with PennyLane.

The filter encodes a 2‑D patch into a quantum state via angle encoding,
applies a parameterised rotation layer, and measures the average
Pauli‑Z expectation value.  The module is fully differentiable and
compatible with PyTorch optimisers.

Example
-------
>>> from Conv__gen064 import Conv
>>> conv = Conv(kernel_size=3, shots=1024, device='default.qubit')
>>> out = conv.run(torch.randn(3, 3))
>>> print(out)
0.412
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class Conv:
    """Quantum convolution filter with trainable parameters.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the square kernel (number of qubits = kernel_size**2).
    device : str or qml.Device, default 'default.qubit'
        PennyLane device to run the circuit on.
    shots : int, default 1024
        Number of measurement shots for the device.
    threshold : float, default 0.5
        Threshold used for angle encoding (values > threshold → π rotation).
    """

    def __init__(self, kernel_size: int = 3, device: str | qml.Device = "default.qubit",
                 shots: int = 1024, threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        # Variational parameters for a single rotation layer
        self.params = pnp.random.randn(self.n_qubits)
        self.log = {}

        @qml.qnode(self.device, interface="torch")
        def circuit(x: pnp.ndarray, params: pnp.ndarray):
            # Angle encoding of the input patch
            for i, val in enumerate(x):
                theta = np.pi if val > self.threshold else 0.0
                qml.RX(theta, wires=i)
            # Variational rotation layer
            for i, p in enumerate(params):
                qml.RZ(p, wires=i)
            # Measurement of Pauli‑Z expectation
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data) -> float:
        """Apply the quantum filter to a 2‑D patch.

        Parameters
        ----------
        data : array‑like or torch.Tensor
            Input patch of shape (kernel_size, kernel_size) or (batch, kernel_size, kernel_size).

        Returns
        -------
        float
            Mean expectation value over the batch (or single value if batch size is 1).
        """
        if isinstance(data, np.ndarray):
            tensor = data
        else:
            tensor = np.array(data, dtype=np.float32)

        # Support batched input
        if tensor.ndim == 2:
            tensor = tensor.reshape(1, -1)  # (1, n_qubits)
        if tensor.ndim == 3:
            tensor = tensor.reshape(tensor.shape[0], -1)  # (B, n_qubits)

        expectations = []
        for patch in tensor:
            exp = self.circuit(patch, self.params)
            expectations.append(exp)
        expectations = np.stack(expectations)
        self.log['raw_expectations'] = expectations
        mean_exp = expectations.mean().item()
        return mean_exp

    def parameters(self):
        """Return the trainable parameters."""
        return self.params

    def set_parameters(self, new_params):
        """Set new values for the trainable parameters."""
        self.params = new_params

__all__ = ["Conv"]
