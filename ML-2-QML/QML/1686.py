"""ConvEnhancedQ: A hybrid variational circuit that emulates the classical filter and provides a learnable quantum back‑end.

The class implements a parameterised circuit whose rotation angles are
controlled by the classical kernel weights. A simple classical‑to‑quantum
mapping is used: each weight is encoded as a rotation around X.  The
circuit is executed on a simulator and returns the average probability of
measuring |1> over all qubits.  The returned value is used as a scalar
gradient contribution in the classical ConvEnhanced module.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import Aer, execute
from typing import Iterable, List


class ConvEnhancedQ:
    """
    Variational quanvolution that can be attached to ConvEnhanced.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 1024):
        """
        :param kernel_size: size of the 2‑D kernel.
        :param threshold: threshold for binarising the input patch.
        :param shots: number of shots for the simulator.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Create a parameterised circuit with one RX rotation per qubit.
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self._circuit.rx(self.theta[0], 0)
        for i in range(1, self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        # Add a simple entangling layer to capture correlations.
        self._circuit.h(0)
        self._circuit.cx(0, 1)
        self._circuit.barrier()
        self._circuit.measure_all()

    def run(self, data_patch: np.ndarray) -> float:
        """
        Execute the circuit on a single data patch.

        :param data_patch: 2D array of shape (kernel_size, kernel_size).
        :return: mean probability of measuring |1> over all qubits.
        """
        # Encode the patch by setting the angle of each RX gate
        param_binds = {}
        for i, val in enumerate(data_patch.flatten()):
            param_binds[self.theta[i]] = np.pi if val > self.threshold else 0.0

        job = execute(self._circuit, self.backend, shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute mean |1> probability
        total_ones = 0
        for bitstring, c in counts.items():
            total_ones += sum(int(b) for b in bitstring) * c
        mean_prob = total_ones / (self.shots * self.n_qubits)
        return mean_prob

    def __call__(self, data_patch: np.ndarray) -> float:
        """
        Make the instance callable so it can be passed as a quantum_back
        to ConvEnhanced.
        """
        return self.run(data_patch)

__all__ = ["ConvEnhancedQ"]
