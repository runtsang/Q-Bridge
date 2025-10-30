import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class AutoencoderGen280(SamplerQNN):
    """
    Quantum autoencoder that returns a 2‑dimensional probability vector.
    The circuit consists of a RealAmplitudes ansatz followed by a SWAP test
    that entangles a latent sub‑space with a trash sub‑space, mirroring the
    original design but with a clean, reusable interface.
    """
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        backend=None,
    ):
        circuit = self._build_circuit(num_latent, num_trash, reps)
        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=(2,),
            sampler=Sampler(backend=backend) if backend else Sampler(),
        )

    def _build_circuit(self, num_latent, num_trash, reps):
        """
        Builds a variational circuit with a RealAmplitudes ansatz and a
        SWAP‑test based latent extraction.  The circuit is compatible with
        :class:`qiskit_machine_learning.neural_networks.SamplerQNN` and
        can be used directly as a torch.nn.Module.
        """
        total_qubits = num_latent + 2 * num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Ansatz on the first (latent + trash) qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
        circuit.compose(ansatz, range(num_latent + num_trash), inplace=True)
        circuit.barrier()

        aux = num_latent + 2 * num_trash  # auxiliary qubit index
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

__all__ = ["AutoencoderGen280"]
