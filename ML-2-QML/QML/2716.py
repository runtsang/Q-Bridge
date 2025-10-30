import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Simple parameterized quantum circuit that returns the expectation of Z."""
    def __init__(self, backend=None, shots=1024):
        self.backend = backend or AerSimulator()
        self.shots = shots

    def run(self, inputs):
        """Execute the circuit for each input and return expectation values."""
        results = []
        for val in inputs:
            circ = QuantumCircuit(1)
            circ.h(0)
            circ.ry(val, 0)
            circ.measure_all()
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                z = 1 if bitstring == '0' else -1
                exp += z * count
            exp /= self.shots
            results.append(exp)
        return results

class UnifiedQCNNHybrid:
    """Quantum wrapper exposing a run method compatible with the classical hybrid head."""
    def __init__(self, shots=1024):
        self.circuit = QuantumCircuitWrapper(shots=shots)

    def run(self, inputs):
        return self.circuit.run(inputs)

__all__ = ["QuantumCircuitWrapper", "UnifiedQCNNHybrid"]
