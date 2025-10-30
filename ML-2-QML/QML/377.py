"""Quantum convolutional filter with parameter‑shift gradient estimation.

The QML counterpart implements a parameterised quantum circuit that
acts as a filter over a 2×2 pixel patch.  It provides
* a run() method that returns the expectation value of a measurement
  operator (average |1⟩ probability),
* a gradient() method that estimates the gradient w.r.t. each parameter
  using the parameter‑shift rule, and
* a simple measurement‑error mitigation routine that re‑weights counts
  based on a calibration matrix.
"""

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.providers.aer import AerSimulator


class ConvFilter:
    """Quantum convolutional filter for 2×2 patches.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the patch (kernel_size×kernel_size).
    shots : int, default 1024
        Number of shots per circuit execution.
    threshold : float, default 0.5
        Classical threshold for mapping pixel values to rotation angles.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
        *,
        backend: AerSimulator | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build parametric circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
        self.params = ParameterVector("θ", self.n_qubits)

        for i, p in enumerate(self.params):
            self.circuit.rx(p, i)

        # Add a simple entangling layer (two‑qubit CNOTs in a ring)
        for i in range(self.n_qubits):
            self.circuit.cx(i, (i + 1) % self.n_qubits)

        self.circuit.measure_all()

    def _bind_params(self, data: np.ndarray) -> list[dict[Parameter, float]]:
        """Create a list of parameter bindings for a batch of data."""
        bindings = []
        for patch in data:
            # Map each pixel to an angle: 0→0, >threshold→π
            angles = np.where(patch > self.threshold, np.pi, 0.0)
            binding = {p: a for p, a in zip(self.params, angles)}
            bindings.append(binding)
        return bindings

    def run(self, data: np.ndarray) -> float:
        """Evaluate the filter on a batch of 2×2 patches.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (batch, kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits and
            all shots.
        """
        batch = data.reshape(data.shape[0], -1)
        bindings = self._bind_params(batch)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=bindings,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average |1⟩ probability over all qubits
        total_ones = 0
        total_shots = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_shots += freq

        return total_ones / (total_shots * self.n_qubits)

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """Estimate the gradient of run() w.r.t. each circuit parameter
        using the parameter‑shift rule.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (batch, kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits,).
        """
        shift = np.pi / 2
        grads = np.zeros(self.n_qubits)

        for i in range(self.n_qubits):
            # Shift parameter i by +π/2 and -π/2
            f_plus = self._eval_shifted(data, i, +shift)
            f_minus = self._eval_shifted(data, i, -shift)
            grads[i] = 0.5 * (f_plus - f_minus)

        return grads

    def _eval_shifted(self, data: np.ndarray, idx: int, shift: float) -> float:
        """Run the circuit with a single parameter shifted by `shift`."""
        batch = data.reshape(data.shape[0], -1)
        bindings = []
        for patch in batch:
            angles = np.where(patch > self.threshold, np.pi, 0.0)
            angles[idx] += shift
            binding = {p: a for p, a in zip(self.params, angles)}
            bindings.append(binding)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=bindings,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        total_shots = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_shots += freq

        return total_ones / (total_shots * self.n_qubits)

    def error_mitigation(self, data: np.ndarray) -> float:
        """Apply a very simple measurement‑error mitigation by re‑weighting
        counts with a calibration matrix estimated from a single‑qubit
        identity circuit."""
        # Calibrate single‑qubit measurement errors
        calib = np.zeros((2, 2))
        for bit in [0, 1]:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.measure(0, 0)
            job = execute(circ, backend=self.backend, shots=self.shots)
            counts = job.result().get_counts(circ)
            calib[bit, 0] = counts.get("0", 0) / self.shots
            calib[bit, 1] = counts.get("1", 0) / self.shots

        # Invert calibration matrix
        inv_calib = np.linalg.inv(calib)

        # Run the original circuit
        raw = self.run(data)

        # Re‑weight the raw expectation value
        factor = inv_calib[1, 1]
        bias = inv_calib[1, 0]
        return raw * factor + bias

def Conv() -> ConvFilter:
    """Convenience factory returning a default ConvFilter instance."""
    return ConvFilter(kernel_size=2, shots=1024, threshold=0.5)
