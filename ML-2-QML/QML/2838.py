import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from typing import Iterable

class HybridFCL:
    """Hybrid fully connected layer using a quantum circuit and sampler."""
    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circ = qiskit.QuantumCircuit(n_qubits)
        self.circ.h(range(n_qubits))
        self.circ.barrier()
        self.circ.ry(self.theta, range(n_qubits))
        self.circ.measure_all()
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = qiskit.QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        self.sampler_qnn = SamplerQNN(
            circuit=qc2,
            input_params=inputs2,
            weight_params=weights2,
            sampler=StatevectorSampler()
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the quantum circuit with the provided thetas and
        return the expectation value.  If the sampler is used,
        its probabilities are combined with the expectation.
        """
        job = qiskit.execute(
            self.circ,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circ)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        sampler_output = self.sampler_qnn.forward(
            {"input": thetas[:2], "weight": thetas[2:6]}
        ).data.numpy()
        combined = expectation + sampler_output.sum() * 0.1
        return np.array([combined])

def FCL() -> HybridFCL:
    """Factory function returning a HybridFCL instance."""
    return HybridFCL()

__all__ = ["HybridFCL", "FCL"]
