"""ConvEnhanced: Variational quanvolution layer with trainable per‑qubit angles."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import SPSA

class ConvEnhanced:
    """
    Variational quanvolution layer that learns a set of rotation angles
    for each qubit.  The circuit encodes a 2‑D patch of size ``kernel_size``
    by applying a rotation RY(theta_i * x_i) to each qubit, where ``x_i``
    is the input pixel value.  A simple entangling block follows, and the
    layer outputs the average probability of measuring |1> over all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        learning_rate: float = 0.01,
        epochs: int = 200,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Trainable parameters: one angle per qubit
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]

        # Build a template circuit with placeholders for data encoding
        self._circuit_template = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit_template.ry(self.theta[i], i)
        # Simple entangling block
        for i in range(self.n_qubits - 1):
            self._circuit_template.cx(i, i + 1)
        self._circuit_template.measure_all()

    def _build_circuit(self, data: np.ndarray) -> QuantumCircuit:
        """
        Create a circuit instance with the data encoded into rotation angles.
        ``data`` must be a 1‑D array of length ``n_qubits`` containing pixel values.
        """
        circ = self._circuit_template.copy()
        bind_dict = {}
        for i, val in enumerate(data.flat):
            # Scale pixel to [0,π] and multiply by trainable theta
            bind_dict[self.theta[i]] = val * np.pi
        circ = circ.bind_parameters(bind_dict)
        return circ

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on the provided ``data`` patch and return the
        average probability of measuring |1> across all qubits.
        """
        circ = self._build_circuit(data)
        job = execute(circ, self.backend, shots=self.shots)
        result = job.result().get_counts(circ)
        # Convert counts to probability of |1> per qubit
        prob_one = 0.0
        for bitstring, count in result.items():
            ones = bitstring.count("1")
            prob_one += ones * count
        return prob_one / (self.shots * self.n_qubits)

    def train(
        self,
        dataset: list[tuple[np.ndarray, float]],
        loss_fn=lambda pred, target: (pred - target) ** 2,
    ) -> None:
        """
        Light‑weight training loop that optimizes the rotation angles
        using the SPSA optimizer.  ``dataset`` is a list of (patch, target)
        tuples where ``patch`` is a 2‑D NumPy array of shape
        (kernel_size, kernel_size) and ``target`` is a scalar.
        """
        # Initialize parameters randomly
        params = np.random.uniform(0, 2 * np.pi, size=self.n_qubits)
        optimizer = SPSA(maxiter=self.epochs, learning_rate=self.learning_rate)

        def objective(p):
            # Update circuit parameters
            for i, val in enumerate(p):
                self.theta[i].assign(val)
            loss = 0.0
            for patch, target in dataset:
                pred = self.run(patch)
                loss += loss_fn(pred, target)
            return loss / len(dataset)

        optimizer.optimize(num_vars=self.n_qubits, objective_function=objective, initial_point=params)

    def predict(self, data: np.ndarray) -> float:
        """
        Alias for :meth:`run`.  Provided for consistency with the ML counterpart.
        """
        return self.run(data)

__all__ = ["ConvEnhanced"]
