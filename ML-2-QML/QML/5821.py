"""Quantum autoencoder using Qiskit's SamplerQNN."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def Autoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """
    Returns a :class:`SamplerQNN` representing a quantum autoencoder circuit.
    
    The circuit consists of:
      * A parameterized RealAmplitudes ansatz on the latent + trash qubits.
      * A simple entangling layer (CX gates) between latent and trash qubits.
      * A swap test with an auxiliary qubit to provide a measurement‑based decoder.
    
    Parameters
    ----------
    num_latent : int
        Number of latent qubits.
    num_trash : int
        Number of trash qubits used for the swap test.
    
    Returns
    -------
    SamplerQNN
        A variational quantum neural network ready to be integrated into a hybrid pipeline.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        """Parameterized ansatz for the latent + trash subspace."""
        qc = QuantumCircuit(num_qubits)
        qc.append(RealAmplitudes(num_qubits, reps=5), range(num_qubits))
        return qc

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        """Builds the complete autoencoder circuit."""
        total_qubits = num_latent + 2 * num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode phase
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)

        # Entangling layer
        for i in range(num_trash):
            qc.cx(num_latent + i, num_latent + num_trash + i)

        qc.barrier()

        # Swap test for measurement‑based decoding
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    circuit = auto_encoder_circuit(num_latent, num_trash)

    def interpret(x: np.ndarray) -> np.ndarray:
        """Interpret the sampler output as a probability distribution."""
        return x.reshape(-1)

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=interpret,
        output_shape=(2,),
        sampler=sampler,
    )
    return qnn


__all__ = ["Autoencoder"]
