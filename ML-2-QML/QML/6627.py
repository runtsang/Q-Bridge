"""Quantum classifier that can be used with the classical HybridConv.

The module implements a parameterized ansatz with explicit data encoding and a depth‑controlled
layer of variational rotations.  Measurements of Z‑observables on each qubit provide
expectation values that are used as classification logits.  The circuit is built using
Qiskit and executed on a chosen backend.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

class HybridConv:
    """
    Quantum counterpart to the classical HybridConv.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (length of the flattened feature map).
    depth : int
        Number of variational layers.
    backend : qiskit.providers.Backend, optional
        Quantum backend; defaults to Aer qasm simulator.
    shots : int, default=1024
        Number of shots per execution.
    threshold : float, default=0.5
        Threshold used to encode classical bits into rotation angles.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding_params, self.var_params, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a layered ansatz with data encoding and variational parameters.
        """
        # Data encoding via RX rotations
        encoding = ParameterVector("x", self.num_qubits)
        # Variational parameters
        var_params = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(var_params[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        circuit.measure_all()

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circuit, encoding, var_params, observables

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on classical data and return a scalar.

        Parameters
        ----------
        data : np.ndarray
            1‑D array of length `num_qubits` containing feature values.

        Returns
        -------
        float
            Normalised expectation value of the Z observables (average over qubits).
        """
        if data.ndim!= 1 or data.size!= self.num_qubits:
            raise ValueError(f"Input data must be 1‑D array of length {self.num_qubits}")

        # Bind encoding parameters based on threshold
        param_binds = []
        for val in data:
            bind = {}
            for i, theta in enumerate(self.encoding_params):
                bind[theta] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        # Execute
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average number of |1⟩ across all qubits
        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += bitstring.count("1") * freq

        avg_probability = total_ones / (self.shots * self.num_qubits)
        return avg_probability

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_qubits={self.num_qubits}, depth={self.depth}, "
            f"shots={self.shots}, threshold={self.threshold})"
)

__all__ = ["HybridConv"]
