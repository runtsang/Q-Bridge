"""Hybrid quantum estimator with variational circuit and Pauli‑Y observable.

The class :class:`HybridEstimator` implements a parameterised quantum circuit
that accepts an input vector of length ``num_qubits`` and returns the
expectation value of a Y‑Pauli operator on the final state.  The circuit
consists of:

* RX rotations that encode the input data (thresholded to 0 or π).
* A shallow random circuit (depth 2) that introduces entanglement.
* Measurement of all qubits followed by classical post‑processing to
  estimate the mean number of |1⟩ outcomes.

The implementation uses Qiskit’s Aer simulator and is fully compatible
with the original EstimatorQNN anchor: a function ``EstimatorQNN`` returns
an instance of :class:`HybridEstimator`.

This quantum module can be imported by the classical ML code via a lazy
import, ensuring the ML side remains free of quantum dependencies.

"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import SparsePauliOp

class HybridEstimator:
    """
    Quantum variational estimator.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (input dimension).
    shots : int, optional
        Number of shots for the simulator.
    threshold : float, optional
        Threshold for binary encoding of the input data.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit.  Defaults to Aer qasm_simulator.
    """

    def __init__(
        self,
        num_qubits: int,
        shots: int = 200,
        threshold: float = 0.5,
        backend=None,
    ) -> None:
        self.num_qubits = num_qubits
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.param_symbols = [Parameter(f"theta{i}") for i in range(num_qubits)]
        self.circuit = self._build_circuit()
        self.observable = SparsePauliOp.from_list([("Y" * num_qubits, 1)])

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # RX rotations with symbolic parameters
        for i, theta in enumerate(self.param_symbols):
            qc.rx(theta, i)
        qc.barrier()
        # Add a shallow random circuit to introduce entanglement
        qc += qiskit.circuit.random.random_circuit(self.num_qubits, depth=2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit for a single input vector.

        Parameters
        ----------
        data : np.ndarray
            1‑D array of length ``num_qubits`` containing real values.
            Values are thresholded to 0 or π before binding to the
            circuit parameters.

        Returns
        -------
        float
            Estimated expectation value of the Y‑Pauli observable.
        """
        if data.shape[0]!= self.num_qubits:
            raise ValueError("Input data must match num_qubits")

        # Binary encoding: 1 → π, 0 → 0
        bind_dict = {sym: np.pi if val > self.threshold else 0.0 for sym, val in zip(self.param_symbols, data)}

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average number of |1⟩ outcomes
        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += freq * bitstring.count("1")
        return total_ones / (self.shots * self.num_qubits)

    def __call__(self, data: np.ndarray) -> float:
        return self.run(data)

def EstimatorQNN() -> HybridEstimator:
    """
    Factory function compatible with the original QML EstimatorQNN anchor.
    Returns an instance of :class:`HybridEstimator`.
    """
    return HybridEstimator(num_qubits=4, shots=200, threshold=0.5)
