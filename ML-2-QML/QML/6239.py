from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import Statevector

def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply a domain‑wall (X gates) to qubits in the range [start, end)."""
    for i in range(start, end):
        circuit.x(i)
    return circuit

class EstimatorQNN__gen044:
    """
    Quantum hybrid model: a variational auto‑encoder that outputs a latent state,
    followed by a quantum estimator that maps the latent to a scalar expectation.
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 5) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.circuit = self._build_circuit()
        self.estimator = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x[0],
            output_shape=1,
            sampler=Sampler(),
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational auto‑encoder ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()

        # Swap‑test style latent extraction
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)

        # Optional domain‑wall to encode prior knowledge
        qc = _domain_wall(qc, 0, total_qubits)

        qc.measure(aux, cr[0])
        return qc

    def __call__(self, params: list[float]) -> list[float]:
        """
        Run the quantum estimator with the provided weight parameters.
        The parameters correspond to the variational angles of the auto‑encoder.
        """
        return self.estimator.run(params)

__all__ = ["EstimatorQNN__gen044"]
