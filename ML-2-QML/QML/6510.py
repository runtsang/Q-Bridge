from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridSamplerAutoEncoder:
    """
    Quantum implementation of a hybrid sampler that mirrors the classical
    HybridSamplerAutoEncoder. The circuit encodes a 2‑dimensional classical
    feature vector into a set of input parameters, applies a RealAmplitudes
    ansatz, and uses a swap‑test with a domain‑wall ancillary register to
    compare the encoded state with a latent reference. The output of the
    StatevectorSampler is interpreted as a 2‑dimensional probability
    distribution that can be directly compared to the classical softmax
    output.
    """
    def __init__(self, latent_dim: int = 4, reps: int = 5):
        self.latent_dim = latent_dim
        self.reps = reps

    def _domain_wall(self, qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        """Apply an X‑flip on a contiguous block of qubits to create a domain wall."""
        for i in range(start, end):
            qc.x(i)
        return qc

    def _swap_test_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        """Build the swap‑test sub‑circuit used for measuring similarity."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, name="q")
        cr = ClassicalRegister(1, name="c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz block
        ansatz = RealAmplitudes(num_latent + num_trash, reps=self.reps)
        qc.append(ansatz, range(num_latent + num_trash))

        qc.barrier()

        # Domain wall on the trash qubits
        qc = self._domain_wall(qc, num_latent, num_latent + num_trash)

        # Swap test
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def build(self) -> SamplerQNN:
        """Instantiate a SamplerQNN that can be trained on classical data."""
        # Parameter vectors
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", self.latent_dim * 2)  # arbitrary weight count

        circuit = self._swap_test_circuit(self.latent_dim, 2)
        qnn = SamplerQNN(
            circuit=circuit,
            input_params=inputs,
            weight_params=weights,
            interpret=lambda x: x,  # identity interpret
            output_shape=2,
            sampler=StatevectorSampler()
        )
        return qnn
