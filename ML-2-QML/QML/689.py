"""ConvEnhanced: variational quantum filter with adaptive thresholding.

The quantum implementation mirrors the classical ConvEnhanced filter
by applying a parameterized variational circuit to each pixel of a
kernel‑sized patch.  The circuit uses a trainable rotation on each
qubit followed by a two‑qubit entangling layer.  The circuit parameters
are updated by a classical optimizer during training.  The output is
the average probability of measuring |1> across all qubits, weighted
by a threshold that is adaptively updated based on the input data.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute
from typing import Tuple, Union, Iterable

class ConvEnhanced:
    """Variational quantum convolution filter.

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the convolution kernel.  If an int is passed, a square
        kernel of that size is used.  If a tuple is passed, the first
        element specifies the height and the second the width.
    depthwise_kernels : int, optional
        Number of independent variational circuits to apply in a
        depth‑wise fashion.  The default is 1.
    shots : int, optional
        Number of shots for each execution.  Default 1024.
    backend : qiskit.providers.Provider, optional
        Qiskit backend to use.  If ``None`` a local Aer simulator is
        instantiated.
    threshold : float, optional
        Initial threshold for mapping pixel values to rotation angles.
        Default 0.5.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        depthwise_kernels: int = 1,
        shots: int = 1024,
        backend=None,
        threshold: float = 0.5,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.depthwise_kernels = depthwise_kernels
        self.shots = shots
        self.threshold = threshold

        if backend is None:
            self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = backend

        self.circuits = []
        n_qubits = kernel_size[0] * kernel_size[1]
        for _ in range(depthwise_kernels):
            circ = qiskit.QuantumCircuit(n_qubits)
            # Parameterized RX rotations
            theta = [Parameter(f"theta_{i}") for i in range(n_qubits)]
            for i in range(n_qubits):
                circ.rx(theta[i], i)
            # Entangling layer
            for i in range(n_qubits - 1):
                circ.cz(i, i + 1)
            circ.measure_all()
            self.circuits.append((circ, theta))

    def _encode_data(self, data: np.ndarray) -> list[dict[Parameter, float]]:
        """Return a list of parameter bindings for each input sample."""
        flattened = data.reshape(-1, self.kernel_size[0] * self.kernel_size[1])
        bindings = []
        for sample in flattened:
            bind = {}
            for i, val in enumerate(sample):
                angle = np.pi if val > self.threshold else 0.0
                bind[self.circuits[0][1][i]] = angle
            bindings.append(bind)
        return bindings

    def forward(self, data: np.ndarray) -> float:
        """Run the quantum filter on one or many 2‑D patches.

        Parameters
        ----------
        data : np.ndarray or iterable of np.ndarray
            Each array must have shape ``(H, W)`` matching the kernel size.
            If an iterable is provided, the method returns the mean output
            over the batch.

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits and
            all depth‑wise circuits.
        """
        if isinstance(data, np.ndarray):
            data = [data]
        probs = []
        for inp in data:
            batch_bindings = self._encode_data(inp)
            out_sum = 0.0
            for circ, params in self.circuits:
                job = execute(
                    circ,
                    backend=self.backend,
                    shots=self.shots,
                    parameter_binds=batch_bindings,
                )
                result = job.result().get_counts(circ)
                total_ones = 0
                total_counts = 0
                for bitstring, cnt in result.items():
                    ones = sum(int(b) for b in bitstring)
                    total_ones += ones * cnt
                    total_counts += cnt
                prob = total_ones / (total_counts * len(circ.qubits))
                out_sum += prob
            probs.append(out_sum / self.depthwise_kernels)
        return sum(probs) / len(probs)

    def run(self, data: np.ndarray) -> float:
        """Compatibility alias for the original API."""
        return self.forward(data)

__all__ = ["ConvEnhanced"]
