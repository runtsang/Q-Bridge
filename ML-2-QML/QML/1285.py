"""Enhanced quantum sampler network with 3‑qubit ansatz and sampling interface."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from typing import Tuple

class SamplerQNNExtended:
    """
    Quantum sampler with:
    - 3‑qubit parameterised ansatz.
    - Two layers of entangling gates.
    - Dedicated input and weight parameters.
    - `sample` method returning classical samples from the quantum state.
    """
    def __init__(self, circuit: QuantumCircuit, input_params: ParameterVector, weight_params: ParameterVector, sampler: QiskitSampler) -> None:
        self.circuit = circuit
        self.input_params = input_params
        self.weight_params = weight_params
        self.sampler = sampler

    def sample(self, inputs: Tuple[float, float, float], num_shots: int = 1024) -> Tuple[int,...]:
        """
        Evaluate the circuit with given inputs and return sampled measurement outcomes.
        """
        bound_circuit = self.circuit.bind_parameters(
            {p: v for p, v in zip(self.input_params, inputs)}
            | {p: v for p, v in zip(self.weight_params, [0.0] * len(self.weight_params))}
        )
        result = self.sampler.run(bound_circuit, shots=num_shots).result()
        counts = result.get_counts()
        samples = []
        for outcome, freq in counts.items():
            samples.extend([int(outcome, 2)] * freq)
        return tuple(samples)

def SamplerQNN() -> SamplerQNNExtended:
    """
    Construct a 3‑qubit variational sampler and return an instance.
    """
    # Parameter vectors
    inputs = ParameterVector("input", 3)
    weights = ParameterVector("weight", 6)

    qc = QuantumCircuit(3)
    # Input rotations
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.ry(inputs[2], 2)

    # Entangling layer 1
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Parameterised rotations
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.ry(weights[2], 2)

    # Entangling layer 2
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Second set of rotations
    qc.ry(weights[3], 0)
    qc.ry(weights[4], 1)
    qc.ry(weights[5], 2)

    # Define sampler primitive
    sampler = QiskitSampler()
    # Wrap into Qiskit neural network interface
    qiskit_sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    # Expose as our extended class
    return SamplerQNNExtended(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )

__all__ = ["SamplerQNN", "SamplerQNNExtended"]
