import warnings
import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def Autoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    reps: int = 5,
    backend_name: str = "qasm_simulator",
    seed: int = 42,
) -> SamplerQNN:
    """
    Returns a variational autoencoder implemented with a RealAmplitudes ansatz.
    The circuit encodes a latent vector into a quantum state, performs a
    swap‑test style comparison with a trash register, and measures the
    auxiliary qubit.  The output is a 2‑dimensional probability vector
    that can be interpreted as a reconstruction loss.

    Parameters
    ----------
    num_latent : int
        Size of the latent space.
    num_trash : int
        Number of ancillary qubits used as a “trash” register.
    reps : int
        Number of repetitions of the RealAmplitudes ansatz.
    backend_name : str
        Qiskit backend name used for sampling.
    seed : int
        Random seed for reproducibility.
    """
    algorithm_globals.random_seed = seed
    qinst = QuantumInstance(
        backend_name,
        shots=1024,
        seed_simulator=seed,
        seed_transpiler=seed,
    )
    sampler = Sampler(quantum_instance=qinst)

    def ansatz(num_qubits: int) -> QuantumCircuit:
        """Variational ansatz for the latent sub‑space."""
        return RealAmplitudes(num_qubits, reps=reps)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # encode latent state
        circuit.compose(
            ansatz(num_latent + num_trash),
            range(0, num_latent + num_trash),
            inplace=True,
        )
        circuit.barrier()

        # swap test with trash qubits
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)

        # measurement
        circuit.measure(aux, cr[0])
        return circuit

    circuit = auto_encoder_circuit(num_latent, num_trash)

    # Optional domain‑wall feature: flip a range of qubits
    def domain_wall(circ: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        for i in range(start, end):
            circ.x(i)
        return circ

    circuit = domain_wall(circuit, 0, num_latent + num_trash)

    # Interpret raw measurement probabilities as a reconstruction vector
    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=(2,),
        sampler=sampler,
    )
    return qnn
