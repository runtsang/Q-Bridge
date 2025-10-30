from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

class HybridEstimatorQNN:
    """
    Quantum counterpart of the HybridEstimatorQNN.  Builds a parameterized
    circuit that maps a classical input vector to a single rotation
    parameter, measures a global Pauli‑Y observable, and returns the
    expectation value.  The API mirrors the classical version.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        backend: str | None = None,
        shots: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or "qasm_simulator"
        self.shots = shots

        # Variational circuit
        self.params = [Parameter(f"θ_{i}") for i in range(n_qubits)]
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        self.circuit.measure_all()

        # Observable: product of Pauli Y on all qubits
        self.observables = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

        # EstimatorQNN from Qiskit Machine Learning
        self.estimator = QiskitEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.params,
            weight_params=[],
            estimator=self.estimator,
        )

    def run(self, x: np.ndarray) -> np.ndarray:
        """
        Run the quantum circuit on a batch of input vectors.

        Parameters
        ----------
        x : np.ndarray
            1‑D array of real numbers to bind to the rotation parameters.
            Length must match ``self.n_qubits``.

        Returns
        -------
        np.ndarray
            Array of expectation values, one per input sample.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        expectations = []
        for sample in x:
            bound = {p: val for p, val in zip(self.params, sample)}
            if self.shots is None:
                # Exact state‑vector evaluation
                exp = self.estimator_qnn.evaluate(bound)[0]
            else:
                # Sampled evaluation
                job = execute(
                    self.circuit,
                    Aer.get_backend(self.backend),
                    shots=self.shots,
                    parameter_binds=[bound],
                )
                result = job.result()
                counts = result.get_counts(self.circuit)
                probs = np.array(list(counts.values())) / self.shots
                states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
                exp = np.sum(states * probs)
            expectations.append(exp)
        return np.array(expectations)

__all__ = ["HybridEstimatorQNN"]
