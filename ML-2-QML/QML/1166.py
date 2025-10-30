"""
ConvGen103: Parameterised variational quanvolution filter.
"""

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumCircuit


class ConvGen103:
    """
    Variational quanvolution filter that maps a 2‑D kernel to a quantum circuit.
    The circuit consists of a TwoLocal ansatz with Ry rotations whose angles are
    set by the pixel values (thresholded).  The output is the average probability
    of measuring |1> across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        threshold: float = 0.5,
        depth: int = 2,
        entanglement: str = "full",
    ):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Build a parameterised TwoLocal circuit
        self.circuit = TwoLocal(
            num_qubits=self.n_qubits,
            rotation_blocks="ry",
            entanglement_blocks=entanglement,
            entanglement="full",
            reps=depth,
            parameter_prefix="theta",
        )
        # Add measurement
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a 2‑D array of pixel values.
        The pixel values are thresholded to set rotation angles.
        Returns the mean probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1)
        param_bind = {}
        for i, param in enumerate(self.circuit.parameters):
            qubit = i % self.n_qubits
            angle = np.pi if flat[qubit] > self.threshold else 0.0
            param_bind[param] = angle

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        total_shots = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_shots += cnt

        return total_ones / (total_shots * self.n_qubits)

    def train(self, data: np.ndarray, target: float, lr: float = 0.01, epochs: int = 10):
        """
        Very light‑weight training loop that adjusts the rotation angles
        using a simple gradient‑descent update on the parameter‑shift rule.
        """
        flat = data.reshape(-1)
        for _ in range(epochs):
            # Build current parameter binding
            param_bind = {}
            for i, param in enumerate(self.circuit.parameters):
                qubit = i % self.n_qubits
                angle = np.pi if flat[qubit] > self.threshold else 0.0
                param_bind[param] = angle

            out = self.run(data)
            loss = (out - target) ** 2

            grads = {}
            shift = np.pi / 2
            for idx, param in enumerate(self.circuit.parameters):
                # Shift +π/2
                bind_plus = dict(param_bind)
                bind_plus[param] = param_bind[param] + shift
                result_plus = execute(
                    self.circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[bind_plus],
                ).result()
                counts_plus = result_plus.get_counts(self.circuit)
                prob_plus = _prob_one(counts_plus)

                # Shift -π/2
                bind_minus = dict(param_bind)
                bind_minus[param] = param_bind[param] - shift
                result_minus = execute(
                    self.circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[bind_minus],
                ).result()
                counts_minus = result_minus.get_counts(self.circuit)
                prob_minus = _prob_one(counts_minus)

                grads[param] = 0.5 * (prob_plus - prob_minus)

            # Update parameters
            for param, grad in grads.items():
                param_bind[param] -= lr * grad

        # Store the updated parameters
        self.circuit.assign_parameters(param_bind, inplace=True)


def _prob_one(counts: dict) -> float:
    """Helper to compute average probability of measuring |1>."""
    total_ones = 0
    total_counts = 0
    for bitstring, cnt in counts.items():
        ones = bitstring.count("1")
        total_ones += ones * cnt
        total_counts += cnt
    return total_ones / (total_counts * len(bitstring))


def Conv(
    kernel_size: int = 2,
    backend=None,
    shots: int = 100,
    threshold: float = 0.5,
    depth: int = 2,
    entanglement: str = "full",
) -> ConvGen103:
    """Factory function mirroring the original Conv() signature."""
    return ConvGen103(
        kernel_size=kernel_size,
        backend=backend,
        shots=shots,
        threshold=threshold,
        depth=depth,
        entanglement=entanglement,
    )


__all__ = ["ConvGen103", "Conv"]
