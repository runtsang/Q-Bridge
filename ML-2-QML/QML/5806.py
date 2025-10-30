"""
HybridFCLQuantum: variational circuit that models a fully‑connected layer.
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import List

class HybridFCLQuantum:
    """Variational quantum circuit that models a fully‑connected layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    backend : backend, optional
        Qiskit backend; defaults to Aer qasm simulator.
    shots : int, optional
        Number of shots for expectation estimation.
    """
    def __init__(self, num_qubits: int, depth: int = 1,
                 backend=None, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        self.circuit = QuantumCircuit(num_qubits)
        for i, qubit in enumerate(range(num_qubits)):
            self.circuit.rx(self.encoding[i], qubit)
        w_idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[w_idx], qubit)
                w_idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)
        self.circuit.measure_all()

        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

    def run(self, thetas: List[float]) -> np.ndarray:
        """Execute the circuit with a flat list of parameters.

        The first ``num_qubits`` values are encoding parameters, followed by
        ``num_qubits * depth`` variational angles.
        """
        if len(thetas)!= self.num_qubits + len(self.weights):
            raise ValueError(
                f"Expected {self.num_qubits + len(self.weights)} parameters, "
                f"got {len(thetas)}"
            )
        param_bindings = {self.encoding[i]: thetas[i] for i in range(self.num_qubits)}
        param_bindings.update(
            {self.weights[i]: thetas[self.num_qubits + i] for i in range(len(self.weights))}
        )
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[param_bindings])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = sum(counts.values())
        expectation = np.zeros(self.num_qubits)
        for state_str, cnt in counts.items():
            prob = cnt / total
            for idx, bit in enumerate(reversed(state_str)):
                if bit == "1":
                    expectation[idx] += prob
        expectation = 2 * expectation - 1
        return expectation

    def weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters for each layer."""
        return [self.num_qubits] + [self.num_qubits] * self.depth

__all__ = ["HybridFCLQuantum"]
