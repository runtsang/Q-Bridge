from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from typing import Iterable, Tuple

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit data encoding and variational parameters.
    The circuit is structured to match the classical API: it returns the circuit,
    encoding parameters, weight parameters, and a list of observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class QuantumClassifierModel:
    """
    Quantum variant of the classifier that mirrors the classical interface.
    It builds a parameterized circuit, attaches a SamplerQNN, and exposes
    a `forward` method that returns a probability distribution over two classes.
    """
    def __init__(self, num_qubits: int, depth: int, use_sampler: bool = True) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = SamplerQNN(
                circuit=self.circuit,
                input_params=self.encoding,
                weight_params=self.weights,
                sampler=StatevectorSampler()
            )

    def forward(self, x: list[float]) -> dict[str, int]:
        """
        Evaluate the circuit with a given input vector `x` (list of floats).
        Returns a raw measurement outcome dictionary when using the sampler.
        """
        if self.use_sampler:
            return self.sampler.sample(x, shots=1024)
        else:
            # Direct measurement of expectation values
            exp_vals = [self.circuit.expectation_value(obs, x) for obs in self.observables]
            # Map expectation values to probabilities via a simple softmax
            import numpy as np
            logits = np.array(exp_vals)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            return {"0": probs[0], "1": probs[1]}

    def get_encoding(self) -> Iterable:
        return self.encoding

    def get_params(self) -> Iterable:
        return self.weights

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
