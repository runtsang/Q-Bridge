"""Quantum convolutional filter using a variational circuit.

This module implements ConvFilter that runs a quantum circuit on 2‑D data.
"""

import numpy as np
import qiskit

class ConvFilter:
    """Quantum filter that emulates a convolution via a parameterised circuit."""

    def __init__(self, kernel_size=2, backend=None, shots=100, threshold=0.5, depth=2):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend

        # Build a parameterised circuit
        self._circuit = self._build_circuit(depth)

    def _build_circuit(self, depth):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # parameterised single‑qubit rotations
        params = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(params):
            qc.rx(p, i)
        # entangling layers
        for _ in range(depth):
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
            for i in range(self.n_qubits):
                qc.rx(params[i], i)  # reuse same params for simplicity
        qc.measure_all()
        return qc

    def run(self, data):
        """Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data_flat = data.reshape(-1)
        param_binds = {}
        for i, val in enumerate(data_flat):
            param_binds[f"θ{i}"] = np.pi if val > self.threshold else 0.0
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        total = 0
        for bitstring, freq in counts.items():
            total += freq * sum(int(b) for b in bitstring)
        return total / (self.shots * self.n_qubits)
