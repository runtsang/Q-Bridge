"""Quantum variational classifier that mirrors the classical interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz with data re‑uploading and entangling gates.
    The returned tuple matches the signature used by the classical helper.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth * 2)

    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit in range(num_qubits):
        qc.rx(encoding[qubit], qubit)

    idx = 0
    for _ in range(depth):
        # Variational rotations
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            qc.rx(weights[idx + 1], qubit)
            idx += 2

        # Entangling layer
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp.from_list([(("I" * i) + "Z" + ("I" * (num_qubits - i - 1)), 1)])
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables


class HybridClassifier:
    """
    Quantum classifier with the same factory interface as the classical version.
    It provides a convenient `run` method that evaluates the expectation values
    of the observables on a backend of choice.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def run(self, data: List[float], backend: str = "statevector_simulator") -> List[float]:
        """
        Execute the circuit for the supplied data vector and return the
        expectation values of each observable.
        """
        bound = {p: v for p, v in zip(self.encoding, data)}
        bound.update({p: 0.0 for p in self.weights})  # initialise weights to zero
        qc = self.circuit.bind_parameters(bound)

        simulator = AerSimulator()
        job = simulator.run(qc, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        probabilities = {state: count / 1024 for state, count in counts.items()}

        expectations = []
        for obs in self.observables:
            exp = 0.0
            for state, prob in probabilities.items():
                bits = [int(b) for b in state[::-1]]
                parity = sum(
                    bits[i] for i in range(len(bits)) if obs.paulis[i] == "Z"
                ) % 2
                exp += prob * (-1) ** parity
            expectations.append(exp)
        return expectations

    def parameters(self) -> List[ParameterVector]:
        """Return the list of variational parameters."""
        return self.weights

    def estimator_qnn(self) -> QiskitEstimatorQNN:
        """
        Instantiate a qiskit EstimatorQNN that uses the same circuit and observables.
        """
        estimator = StatevectorEstimator()
        return QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=estimator,
        )


def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Small regression circuit mirroring the qiskit EstimatorQNN example.
    """
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    observable1 = SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc1,
        observables=[observable1],
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )


__all__ = ["HybridClassifier", "EstimatorQNN", "build_classifier_circuit"]
