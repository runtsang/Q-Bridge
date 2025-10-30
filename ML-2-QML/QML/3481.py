"""Hybrid quantum sampler that accepts parameters from the classical encoder.

Combines the fully‑connected layer circuit from the FCL seed with
the SamplerQNN style parameterized circuit.
"""

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute

class HybridQuantumSampler:
    """
    Parameterized quantum circuit with separate input and weight parameters.
    The circuit consists of an H‑gate, CX‑entanglement and Ry rotations
    similar to the FCL seed, followed by a SamplerQNN‑style entangling block.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Input and weight parameters
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        # Apply Ry using input parameters
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        # SamplerQNN block
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the given parameters.
        Parameters are expected in the order:
        [input0, input1, weight0, weight1, weight2, weight3].
        Returns the expectation value of the measurement outcome.
        """
        if len(params)!= self.n_qubits + len(self.weights):
            raise ValueError("Parameter vector has incorrect length.")
        bind_dict = {
            self.inputs[0]: params[0],
            self.inputs[1]: params[1],
            self.weights[0]: params[2],
            self.weights[1]: params[3],
            self.weights[2]: params[4],
            self.weights[3]: params[5],
        }
        bound_circ = self.circuit.bind_parameters(bind_dict)
        job = execute(bound_circ, self.backend, shots=self.shots)
        result = job.result().get_counts(bound_circ)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridQuantumSampler"]
