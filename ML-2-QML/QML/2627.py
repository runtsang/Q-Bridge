"""Quantum core for the hybrid classifier: variational circuit, sampler, and observables."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a layered variational ansatz with explicit encoding and variational parameters.
    Returns:
        circuit: QuantumCircuit ready for simulation or transpilation.
        encoding: list of encoding parameters.
        weights: list of variational parameters.
        observables: list of Pauli‑Z observables for feature extraction.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


def SamplerQNN() -> QiskitSamplerQNN:
    """
    A parameterised quantum sampler circuit that can be integrated into the hybrid architecture.
    The sampler outputs a probability distribution over two outcomes.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = Sampler()
    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn


class HybridClassifier:
    """
    Quantum core that provides a variational circuit and a sampler sub‑circuit.
    The class exposes methods to evaluate expectation values and to sample probabilities.
    """

    def __init__(self, num_qubits: int, depth: int):
        self.circuit, self.encoding_params, self.weight_params, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = AerSimulator()

    def evaluate_expectations(self, enc_values: List[float], weight_values: List[float]) -> List[float]:
        """
        Bind encoding and weight parameters, run the circuit, and return expectation values of Z observables.
        """
        param_bindings = {
            self.encoding_params[i]: enc_values[i]
            for i in range(len(self.encoding_params))
        }
        param_bindings.update(
            {self.weight_params[i]: weight_values[i] for i in range(len(self.weight_params))}
        )
        bound = self.circuit.bind_parameters(param_bindings)
        result = self.backend.run(bound).result()
        state = Statevector(result.get_statevector(bound))
        return [state.expectation_value(obs).real for obs in self.observables]

    def sample_probabilities(self, enc_values: List[float], weight_values: List[float]) -> List[float]:
        """
        Use the integrated SamplerQNN to produce a probability distribution over two outcomes.
        """
        sampler_qnn = SamplerQNN()
        param_bindings = {
            sampler_qnn.input_params[0]: enc_values[0],
            sampler_qnn.input_params[1]: enc_values[1],
            sampler_qnn.weight_params[0]: weight_values[0],
            sampler_qnn.weight_params[1]: weight_values[1],
            sampler_qnn.weight_params[2]: weight_values[2],
            sampler_qnn.weight_params[3]: weight_values[3],
        }
        bound = sampler_qnn.circuit.bind_parameters(param_bindings)
        result = self.backend.run(bound).result()
        counts = result.get_counts(bound)
        total = sum(counts.values())
        return [counts.get("0", 0) / total, counts.get("1", 0) / total]


__all__ = ["HybridClassifier", "build_classifier_circuit", "SamplerQNN"]
