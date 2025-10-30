"""ConvGen200 – quantum multi‑scale variational filter.

The class implements a variational circuit for each kernel size
(2,3,4).  Each circuit shares the same structure but has a different
number of qubits.  Data is encoded as rotation angles and the
output is the average probability of measuring |1> across all qubits.

Example::
    from conv_gen200_qml import ConvGen200
    qfilter = ConvGen200()
    out = qfilter.run(np.random.randint(0,256,(3,3)))
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen200"]

class ConvGen200:
    """Variational quantum filter for multiple kernel sizes.

    Parameters
    ----------
    kernel_sizes : list[int] | None
        List of kernel sizes to support.  Defaults to [2, 3, 4].
    backend : qiskit.providers.Backend | None
        Backend to execute circuits on.  Defaults to Aer qasm simulator.
    shots : int
        Number of shots per execution.
    threshold : float
        Data threshold for encoding (pixel > threshold -> pi rotation).
    """
    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        backend=None,
        shots: int = 200,
        threshold: float = 127.0,
    ) -> None:
        self.kernel_sizes = kernel_sizes or [2, 3, 4]
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Create a circuit for each kernel size
        self.circuits = {}
        for ks in self.kernel_sizes:
            n = ks * ks
            qc = QuantumCircuit(n, n)
            theta = [Parameter(f"θ{i}") for i in range(n)]
            # Encode data with RX rotations
            for i in range(n):
                qc.rx(theta[i], i)
            # Add a small entangling layer
            qc += random_circuit(n, 2)
            qc.measure_all()
            self.circuits[ks] = (qc, theta)

    def run(self, data: np.ndarray) -> float:
        """Execute the appropriate circuit for the data shape.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (k, k) where k is one of the supported kernel sizes.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        ks = data.shape[0]
        if ks not in self.circuits:
            raise ValueError(f"Unsupported kernel size {ks}. Supported: {list(self.circuits.keys())}")

        qc, theta = self.circuits[ks]
        # Flatten data and bind parameters
        flat = data.flatten()
        param_binds = {}
        for i, val in enumerate(flat):
            param_binds[theta[i]] = np.pi if val > self.threshold else 0.0

        job = execute(qc, self.backend, shots=self.shots, parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(qc)
        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += sum(int(b) for b in bitstring) * freq
        prob = total_ones / (self.shots * ks * ks)
        return prob
