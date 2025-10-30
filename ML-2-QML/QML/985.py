"""Quantum fully connected layer with parameterized rotation gates."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector
from typing import Iterable, Optional

class QuantumFCLayer:
    """
    Parameterised quantum circuit that mimics a fully connected layer.
    Each input feature corresponds to a qubit that receives an RY(theta)
    rotation. The circuit creates entanglement with a simple chain of CNOTs,
    measures all qubits, and computes the expectation value of the Pauli Z
    operator averaged over all qubits.
    """
    def __init__(self, n_qubits: int = 1, backend: Optional[str] = "qasm_simulator",
                 shots: int = 1024, entangle: bool = True) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.entangle = entangle

        self.backend = AerSimulator() if backend == "qasm_simulator" else backend

        # Parameter vector for RY rotations
        self.theta = ParameterVector("theta", length=n_qubits)

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        if entangle:
            for i in range(n_qubits - 1):
                self.circuit.cx(i, i + 1)
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

        # Transpile for chosen backend
        self.transpiled = transpile(self.circuit, backend=self.backend)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters and return the
        expectation value of Pauli Z averaged over all qubits."""
        bound_circ = self.transpiled.bind_parameters(dict(zip(self.theta, thetas)))
        qobj = assemble(bound_circ, backend=self.backend, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts(bound_circ)
        # Convert bitstrings to integer states
        probs = np.array([counts.get(k, 0) for k in sorted(counts)]) / self.shots
        states = np.array([int(k, 2) for k in sorted(counts)])
        # Compute Pauli-Z expectation per qubit
        exp_z = []
        for i in range(self.n_qubits):
            bits = ((states >> (self.n_qubits - 1 - i)) & 1).astype(int)
            z_vals = 1 - 2 * bits
            exp_z.append(np.sum(z_vals * probs))
        expectation = np.mean(exp_z)
        return np.array([expectation])

def FCL(n_qubits: int = 1, backend: str = "qasm_simulator", shots: int = 1024,
        entangle: bool = True) -> QuantumFCLayer:
    """Factory that returns an instance of the quantum fully connected layer."""
    return QuantumFCLayer(n_qubits, backend, shots, entangle)

__all__ = ["FCL"]
