from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
import numpy as np

class SamplerQNNGen125:
    """
    Quantum SamplerQNN that mirrors the classical SamplerQNNGen125.
    It builds a parameterized circuit combining rotation and entanglement layers,
    then exposes a SamplerQNN instance for sampling.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        num_latent: int = 3,
        num_trash: int = 2,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.shots = shots
        self.backend = backend
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """
        Construct a simple parameterized circuit that includes:
          * Two rotation layers (ry) on each qubit.
          * Entanglement via CNOTs.
          * Additional rotation parameters for weight tuning.
        """
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Parameter vectors
        inputs2 = ParameterVector("input", self.num_qubits)
        weights2 = ParameterVector("weight", 4)

        # Rotation layer
        qc.ry(inputs2[0], 0)
        qc.ry(inputs2[1], 1)
        qc.cx(0, 1)

        # Weight layer
        qc.ry(weights2[0], 0)
        qc.ry(weights2[1], 1)
        qc.cx(0, 1)
        qc.ry(weights2[2], 0)
        qc.ry(weights2[3], 1)

        # Measurement
        qc.measure(qr[0], cr[0])
        return qc

    def run(self, backend=None, shots: int | None = None) -> dict[str, int]:
        """
        Execute the circuit on the specified backend and return measurement counts.
        """
        backend = backend or self.backend
        shots = shots or self.shots
        job = backend.run(self.circuit, shots=shots)
        return job.result().get_counts(self.circuit)

__all__ = ["SamplerQNNGen125"]
