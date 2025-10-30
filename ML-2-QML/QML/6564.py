"""Hybrid convolution + fully‑connected layer implemented with Qiskit.

This quantum implementation mirrors the classical HybridLayer: a parameterised
RX‑based convolution followed by a single‑parameter variational circuit that
serves as the fully‑connected layer.  The ``run(data, thetas)`` method
accepts the same arguments as the classical counterpart.

Usage
-----
>>> from qiskit import Aer
>>> layer = HybridLayer(kernel_size=3, backend=Aer.get_backend("qasm_simulator"))
>>> out = layer.run(data, thetas=[0.5])
"""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class HybridLayer:
    """
    Quantum analogue of the PyTorch HybridLayer.  The circuit contains two
    logical sections:

    * Convolutional section – a set of RX gates whose angles are set to π
      when the corresponding input pixel exceeds ``threshold``.
    * Fully‑connected section – a single variational θ applied to a dedicated
      qubit that encodes the mean of the supplied thetas.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 127,
        n_fc_qubits: int = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits_conv = kernel_size ** 2
        self.n_fc_qubits = n_fc_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._build_circuit()

    def _build_circuit(self) -> None:
        # Total qubits = conv + fc
        self.circuit = qiskit.QuantumCircuit(self.n_qubits_conv + self.n_fc_qubits)
        # Convolutional RX gates
        self.theta_conv = [
            qiskit.circuit.Parameter(f"theta_conv_{i}") for i in range(self.n_qubits_conv)
        ]
        for i in range(self.n_qubits_conv):
            self.circuit.rx(self.theta_conv[i], i)
        # Random circuit layers to entangle the convolution qubits
        self.circuit += random_circuit(self.n_qubits_conv, 2)
        # Fully‑connected variational qubit(s)
        self.theta_fc = qiskit.circuit.Parameter("theta_fc")
        self.circuit.h(range(self.n_qubits_conv, self.n_qubits_conv + self.n_fc_qubits))
        self.circuit.ry(self.theta_fc, range(self.n_qubits_conv, self.n_qubits_conv + self.n_fc_qubits))
        self.circuit.measure_all()

    def run(self, data: np.ndarray, thetas: list[float] | None = None) -> float:
        """
        Execute the hybrid quantum circuit.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).
            thetas: Iterable of floats used to set the variational θ.  If omitted
                    a neutral value of 0.0 is used.

        Returns:
            Expectation value of measuring |1⟩ over all qubits.
        """
        if thetas is None:
            thetas = [0.0]
        # Flatten data to match qubit ordering
        data_flat = np.reshape(data, (1, self.n_qubits_conv))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta_conv[i]] = np.pi if val > self.threshold else 0.0
            bind[self.theta_fc] = thetas[0]
            param_binds.append(bind)
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        # Compute average number of |1⟩ outcomes
        total_ones = 0
        for bitstring, count in result.items():
            total_ones += count * sum(int(bit) for bit in bitstring)
        expectation = total_ones / (self.shots * (self.n_qubits_conv + self.n_fc_qubits))
        return expectation

__all__ = ["HybridLayer"]
