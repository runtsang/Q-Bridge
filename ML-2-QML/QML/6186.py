"""Quantum classifier that mirrors the classical interface and integrates
EstimatorQNN for hybrid training.

The module defines a class `QuantumClassifierModel` that builds a
parameterized circuit using Qiskit.  The circuit is a data‑re‑uploading
ansatz: each input feature is encoded with RX, followed by a series of
variational Ry rotations and CZ entangling gates.  The depth is
controlled by the ``depth`` parameter.  The observables are a set of
Z‑basis measurements on each qubit, matching the classical two‑class
output.  The class exposes a method to return a Qiskit EstimatorQNN
object, which can be trained with Qiskit’s primitives.  It also
provides a convenience ``evaluate`` method that runs the circuit on a
given backend and returns expectation values.

The design follows the “combination” scaling paradigm: increasing
qubit count or depth scales the quantum model, while the classical
model scales by increasing depth or feature dimension.  The shared API
makes it trivial to swap between worlds for benchmarking.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator


class QuantumClassifierModel:
    """
    Quantum implementation of a data‑re‑uploading classifier.

    Parameters
    ----------
    num_qubits: int
        Number of qubits / input features.
    depth: int
        Number of variational layers.
    backend: str | Aer backend
        Backend used for evaluation; defaults to the state‑vector simulator.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Internal routine that constructs the parameterized circuit and returns
        the components needed for EstimatorQNN."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, [encoding], [weights], observables

    def get_estimator_qnn(self) -> EstimatorQNN:
        """Return a Qiskit EstimatorQNN built from the internal circuit."""
        estimator = Estimator(backend=self.backend)
        return EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=estimator,
        )

    def evaluate(self, inputs: Iterable[List[float]]) -> List[List[float]]:
        """
        Run the circuit on the provided inputs and return expectation values.

        Parameters
        ----------
        inputs: Iterable[List[float]]
            A batch of input feature vectors.

        Returns
        -------
        List[List[float]]
            Expectation values for each observable.
        """
        estimator = Estimator(backend=self.backend)
        exp_vals = estimator.run(
            self.circuit,
            [(self.circuit, dict(zip(self.encoding[0], vec))) for vec in inputs],
            self.observables,
        ).values
        return exp_vals

__all__ = ["QuantumClassifierModel"]
