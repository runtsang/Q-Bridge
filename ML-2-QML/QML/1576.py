"""Quantum‑augmented convolutional filter using a variational circuit.

The circuit applies a parameterised RX rotation to each qubit representing a pixel
value and measures all qubits.  The result is the average probability of measuring
|1> across the qubits, which is then passed through a sigmoid activation.

The class is designed to mirror the interface of the classical ConvEnhanced
class, enabling a drop‑in replacement.

Example::

    from Conv__gen138 import ConvEnhanced

    conv_q = ConvEnhanced(kernel_size=3, threshold=0.5, shots=200)
    patch = np.random.rand(3, 3)
    output = conv_q.run(patch)
"""

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal


class ConvEnhanced:
    """
    Quantum convolutional filter.
    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolution kernel.
    threshold : float, default=0.0
        Threshold to decide rotation angle.
    shots : int, default=100
        Number of shots for simulation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size * kernel_size

        # Build a simple variational circuit
        self.circuit = TwoLocal(num_qubits=self.n_qubits,
                                rotation_blocks='ry',
                                entanglement_blocks='cx',
                                entanglement='full',
                                reps=1,
                                insert_barriers=False)
        self.circuit.measure_all()

        self.backend = Aer.get_backend('qasm_simulator')

        # Parameters for each qubit
        self.theta = [Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i, param in enumerate(self.theta):
            self.circuit.add_parameter(param, i)

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a single kernel‑sized patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        # Flatten patch to 1‑D array
        flat = np.reshape(data, (1, self.n_qubits))

        # Bind parameters based on threshold
        param_binds = []
        for dat in flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(self.circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


__all__ = ["ConvEnhanced"]
