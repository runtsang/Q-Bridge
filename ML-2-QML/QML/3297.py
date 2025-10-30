import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoderFCL:
    """
    Quantum auto‑encoder that mirrors the classical hybrid.
    Implements a variational RealAmplitudes ansatz with a swap‑test
    and optional domain‑wall encoding.  The output is a 2‑dimensional
    vector interpreted as a quantum expectation.
    """
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        shots: int = 200,
        reps: int = 5,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.shots = shots
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_autoencoder_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=2,
            sampler=self.sampler,
        )

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _autoencoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode data into first num_latent qubits via ansatz
        circuit.compose(self._ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()

        # Swap‑test with trash qubits
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    def _domain_wall(self, circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
        """Introduce a domain wall (X gates) on qubits a..b-1."""
        for i in range(a, b):
            circuit.x(i)
        return circuit

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the quantum auto‑encoder on a batch of inputs.
        inputs shape is ignored; the circuit uses internal parameters.
        Returns a (batch, 2) array of expectations.
        """
        # For demonstration, we ignore inputs and run the circuit directly.
        # In a full implementation, inputs would be encoded into the circuit.
        result = self.qnn.predict(inputs)
        return result

__all__ = ["HybridAutoencoderFCL"]
