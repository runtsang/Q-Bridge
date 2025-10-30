"""ConvGen101: hybrid variational convolutional circuit.

This class implements a quantum filter that can be trained using
parameter‑shift gradient or any other QGrad method.  It extends the
original quanvolution by adding a trainable variational layer that
maps classical pixel values to rotation angles and applies a
parameterised entangling structure.  The output is a scalar
probability that can be used as a feature in a hybrid model.
"""

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute

class ConvGen101:
    """
    Variational quantum filter for 2‑D convolution.

    Parameters
    ----------
    kernel_size : int
        Size of the convolutional kernel (assumes square).
    backend : qiskit.providers.BaseBackend, optional
        Execution backend.  Defaults to Aer qasm simulator.
    shots : int, optional
        Number of shots per execution.
    threshold : float, optional
        Pixel value threshold used for encoding.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # default backend
        if backend is None:
            self.backend = Aer.get_backend('qasm_simulator')
        else:
            self.backend = backend

        # trainable parameters
        self.params = ParameterVector('theta', self.n_qubits)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        """Construct a parameterised variational circuit."""
        qc = qiskit.QuantumCircuit(self.n_qubits)

        # data‑encoding: rotate each qubit by theta_i
        for i, p in enumerate(self.params):
            qc.rx(p, i)

        # entangling layer: a simple chain of CNOTs
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        # measurement
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a batch of data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> over all qubits.
        """
        # reshape data to 1D vector
        vec = np.reshape(data, (self.n_qubits,))

        # bind parameters: map pixel values to rotation angles
        angle_map = {}
        for i, val in enumerate(vec):
            # map pixel > threshold to pi, else 0
            angle_map[self.params[i]] = np.pi if val > self.threshold else 0.0

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[angle_map])
        result = job.result()
        counts = result.get_counts(self.circuit)

        # compute average probability of |1> across qubits
        total_ones = 0
        total_counts = 0
        for outcome, freq in counts.items():
            ones = sum(int(b) for b in outcome)
            total_ones += ones * freq
            total_counts += freq

        return total_ones / (self.shots * self.n_qubits)

    def get_params(self) -> np.ndarray:
        """Return current parameter values."""
        return np.array([p.value for p in self.params])

    def set_params(self, values: np.ndarray) -> None:
        """Set new parameter values."""
        if len(values)!= len(self.params):
            raise ValueError("Parameter vector length mismatch.")
        for p, val in zip(self.params, values):
            p.assign(val)

__all__ = ["ConvGen101"]
