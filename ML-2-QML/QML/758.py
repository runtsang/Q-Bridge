"""Quantum convolution filter implemented with a parameterized variational circuit.

The class ConvEnhanced mirrors the classical interface: it exposes a
``run`` method that takes a 2‑D array and returns a scalar probability.
The circuit consists of a layer of RX rotations whose angles are
parameterized by learnable parameters, followed by a small entanglement
layer. The expectation value of the Z operator on all qubits is
averaged to produce the filter output. The parameters are stored as a
list of qiskit.circuit.Parameter objects and can be updated by a
classical optimizer.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

__all__ = ["ConvEnhanced"]


class ConvEnhanced:
    """
    Parameterized quantum filter with a depth‑2 entanglement layer.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (determines number of qubits).
    backend : qiskit.providers.BaseBackend
        Backend used for execution (default: Aer qasm simulator).
    shots : int
        Number of shots for sampling.
    threshold : float
        Threshold applied to classical input before encoding.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameter vector for RX rotations
        self.params = ParameterVector("theta", self.n_qubits)

        # Build circuit template
        self.circuit = QuantumCircuit(self.n_qubits)
        # Encode classical data via RX gates
        for i in range(self.n_qubits):
            self.circuit.rx(self.params[i], i)
        # Entanglement layer (nearest‑neighbour CNOTs)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Measurement
        self.circuit.measure_all()

    def _encode_data(self, data: np.ndarray) -> dict:
        """
        Encode classical data into parameter bindings.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        dict
            Mapping from Parameter to float.
        """
        flat = data.flatten()
        bind = {}
        for i, val in enumerate(flat):
            angle = np.pi if val > self.threshold else 0.0
            bind[self.params[i]] = angle
        return bind

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on the provided data and return the mean
        probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of |1> outcome.
        """
        bind = self._encode_data(data)
        bound_circuit = self.circuit.bind_parameters(bind)
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        total = self.shots

        prob_one = []
        for qubit in range(self.n_qubits):
            count_one = sum(
                count for bitstring, count in counts.items() if bitstring[-(qubit + 1)] == "1"
            )
            prob_one.append(count_one / total)

        return np.mean(prob_one)

    def set_parameters(self, new_params: np.ndarray) -> None:
        """
        Update the trainable parameters of the circuit.

        Parameters
        ----------
        new_params : np.ndarray
            Array of shape (n_qubits,) containing new rotation angles.
        """
        if new_params.shape!= (self.n_qubits,):
            raise ValueError("Parameter shape mismatch.")
        # Update the circuit's parameters in place
        for i, val in enumerate(new_params):
            self.circuit.assign_parameters({self.params[i]: val}, inplace=True)
