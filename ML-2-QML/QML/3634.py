"""Hybrid sampler‑classifier using Qiskit."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class HybridSamplerClassifier:
    """Quantum analogue of HybridSamplerClassifier.
    It constructs a sampler circuit that produces a two‑outcome distribution
    and a layered ansatz that acts as a classifier on the sampled bits.
    The interface mirrors the classical implementation for easy comparison."""
    def __init__(self, num_qubits: int = 2, depth: int = 1) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        # Build sampler subcircuit
        inputs = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * 2)
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.ry(inputs[i], i)
        for i in range(num_qubits):
            qc.ry(weights[i], i)
        qc.cx(0, 1)
        # Build classifier subcircuit
        classifier_circ, _, _, _ = self.build_classifier_circuit(num_qubits, depth)
        # Concatenate sampler and classifier
        self.circuit = qc.compose(classifier_circ)
        # Define sampler
        self.sampler = Sampler()
        self.sampler_qnn = SamplerQNN(circuit=self.circuit,
                                      input_params=list(inputs),
                                      weight_params=list(weights),
                                      sampler=self.sampler)

    def run(self, input_vals: List[float]) -> List[float]:
        """Execute the circuit with supplied input parameters and return
        measurement probabilities for each output bit."""
        param_dict = {param: val for param, val in zip(self.sampler_qnn.input_params, input_vals)}
        probs = self.sampler_qnn.run(parameter_values=param_dict, shots=1024)
        return probs

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Construct a layered ansatz with explicit encoding and variational parameters
        matching the classical analogue."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

__all__ = ["HybridSamplerClassifier"]
