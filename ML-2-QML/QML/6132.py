"""ConvGen: a variational quantum filter.

This class mirrors the classical ConvGen but replaces the fixed
convolution with a parameterised quantum circuit.  The circuit
parameters are exposed as a ParameterVector and can be optimised
using Qiskit's gradient utilities.  The circuit is executed on the
Aer simulator and returns the average probability of measuring |1>
across all qubits.
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RandomCircuit

__all__ = ["ConvGen"]


class ConvGen:
    """
    Variational quantum filter that maps a 2‑D patch to a single
    scalar output.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch (and number of qubits).
    threshold : float, default 127
        Value used to encode classical data into quantum parameters.
    shots : int, default 1024
        Number of shots per evaluation.
    backend : qiskit.providers.ProviderBackend, optional
        Backend to run the circuit on.  Defaults to Aer qasm_simulator.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 127,
                 shots: int = 1024,
                 backend=None) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.n_qubits = kernel_size ** 2
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameter vector for data encoding
        theta = ParameterVector('theta', self.n_qubits)
        qc.rx(theta, range(self.n_qubits))
        # Add a random variational layer
        random_layer = RandomCircuit(self.n_qubits, depth=2)
        qc.compose(random_layer, inplace=True)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (kernel_size, kernel_size) with integer values.
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (self.n_qubits,))
        param_bindings = []
        for val in data:
            # Encode each pixel as a rotation angle
            angle = np.pi if val > self.threshold else 0.0
            param_bindings.append({self._circuit.parameters[i]:
                                   angle for i in range(self.n_qubits)})
        job = execute(self._circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=param_bindings)
        result = job.result()
        counts = result.get_counts(self._circuit)
        total = self.shots * self.n_qubits
        ones = sum(int(bit) * count for bit, count in counts.items())
        return ones / total

    def forward(self, data: np.ndarray) -> float:
        return self.run(data)
