import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoderLayer(SamplerQNN):
    """
    Quantum auto‑encoder that implements RealAmplitudes encoding,
    a swap‑test reconstruction, and a fully‑connected quantum layer
    realized as a weighted sum of Pauli‑Z measurements.  It can be
    used interchangeably with the classical HybridAutoencoderLayer
    defined in the ML module.
    """
    def __init__(self,
                 num_latent: int,
                 num_trash: int,
                 shots: int = 1024,
                 backend=None):
        # Build the underlying circuit
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encoding block with RealAmplitudes ansatz
        ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
        circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()

        # Swap‑test reconstruction
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        # Define weight parameters: ansatz params + theta_i for each latent qubit
        weight_params = ansatz.parameters + [qiskit.circuit.Parameter(f"theta_{i}") for i in range(num_latent)]

        # Sampler primitive
        sampler = Sampler(backend=backend) if backend else Sampler()

        # Interpret measurement as weighted sum over Z eigenvalues
        def interpret(x):
            # x is a 1‑D array of measurement results (0/1)
            eig = 2 * x.astype(float) - 1  # map to +/-1
            return np.dot(eig, np.ones(num_latent))  # placeholder: equal weights

        super().__init__(circuit=circuit,
                         input_params=[],
                         weight_params=weight_params,
                         interpret=interpret,
                         output_shape=1,
                         sampler=sampler)
