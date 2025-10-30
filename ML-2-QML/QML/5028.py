import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit import Aer
from typing import Tuple

class HybridFilter:
    """Quantum filter that encodes a patch into a random circuit and measures |1> probabilities."""

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 scaling: Tuple[float, float] = None,
                 shots: int = 1024) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Build the circuit: RX rotations encode data, followed by a random layer
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", length=self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.scaling = scaling or (1.0, 0.0)  # (scale, shift)

    def run(self, data):
        """Execute the circuit on a 2D patch and return a scaled probability."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        avg_prob = counts / (self.shots * self.n_qubits)
        return avg_prob * self.scaling[0] + self.scaling[1]

def Conv() -> HybridFilter:
    """Factory mirroring the original Conv interface for the quantum version."""
    return HybridFilter()

__all__ = ["HybridFilter", "Conv"]
