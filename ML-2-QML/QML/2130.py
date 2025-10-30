"""Quantum convolutional filter with entanglement and parameter‑shift gradient estimation."""
import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute

class ConvEnhancedQ:
    """
    Quantum counterpart of ConvEnhanced.
    Supports an optional entanglement layer and a parameter‑shift
    gradient estimator.  The ``run`` method accepts a 2‑D array
    and returns the average probability of measuring |1> across
    all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        threshold: float = 127.0,
        entanglement: bool = True,
    ):
        """
        Parameters
        ----------
        kernel_size : int
            Size of the filter kernel. Default is 2.
        backend : qiskit.providers.backend.Backend | None
            Qiskit backend to execute the circuit. If None, use Aer's qasm simulator.
        shots : int
            Number of shots per execution. Default is 100.
        threshold : float
            Threshold for converting classical pixel values to rotation angles.
        entanglement : bool
            If True, add a simple CNOT chain entanglement layer.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vector for RX rotations
        self.theta = ParameterVector("theta", self.n_qubits)
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)

        # RX rotations
        self.circuit.rx(self.theta, range(self.n_qubits))

        # Optional entanglement layer
        if entanglement:
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)

        # Measurement
        self.circuit.measure_all()

    def run(self, data) -> float:
        """
        Execute the quantum filter on the input data.

        Parameters
        ----------
        data : numpy.ndarray
            2‑D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.array(data).reshape(-1)
        if data_flat.size!= self.n_qubits:
            raise ValueError(f"Expected data of size {self.n_qubits}, got {data_flat.size}")

        param_bind = {
            self.theta[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(data_flat)
        }

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq

        return total_ones / (self.shots * self.n_qubits)

    def parameter_shift_gradient(self, data, shift=np.pi / 2) -> np.ndarray:
        """
        Estimate the gradient of the output with respect to the RX parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        data : numpy.ndarray
            Input data array.
        shift : float
            Shift amount for the parameter‑shift rule.

        Returns
        -------
        numpy.ndarray
            Gradient vector of shape (n_qubits,).
        """
        grad = np.zeros(self.n_qubits)
        base_output = self.run(data)

        for i in range(self.n_qubits):
            # Positive shift
            pos_bind = self._bind_shift(data, i, shift)
            pos_out = self._run_with_bind(pos_bind)

            # Negative shift
            neg_bind = self._bind_shift(data, i, -shift)
            neg_out = self._run_with_bind(neg_bind)

            grad[i] = (pos_out - neg_out) / (2 * np.sin(shift))

        return grad

    def _bind_shift(self, data, idx, shift):
        data_flat = np.array(data).reshape(-1)
        bind = {
            self.theta[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(data_flat)
        }
        bind[self.theta[idx]] += shift
        return bind

    def _run_with_bind(self, bind):
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
        return total_ones / (self.shots * self.n_qubits)
