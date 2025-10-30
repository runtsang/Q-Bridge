import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def AutoencoderQML():
    """
    Quantum autoencoder that uses a QCNN‑style convolutional ansatz
    and a swap‑test based reconstruction measurement. The circuit
    is constructed to match the hybrid scaling paradigm of the
    classical counterpart.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # --- QCNN‑style convolution block ------------------------------------
    def conv_circuit(params):
        """Two‑qubit convolution unit used in QCNN layers."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def conv_layer(num_qubits, param_prefix):
        """Build a convolutional layer over all qubits in pairs."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(conv_circuit(params[i:i+3]), [i, i+1], inplace=True)
        return qc

    # --- Quantum autoencoder circuit --------------------------------------
    def auto_encoder_circuit(num_latent: int, num_trash: int):
        """
        Construct the full autoencoder.  The latent qubits encode the
        compressed state, the trash qubits are auxiliary for the swap test.
        """
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode: QCNN‑style convolution over latent + first trash block
        qc.compose(conv_layer(num_latent + num_trash, "c"), range(num_latent + num_trash), inplace=True)
        qc.barrier()

        # Swap‑test based reconstruction on the auxiliary qubit
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # Parameters
    num_latent = 3
    num_trash = 2
    circuit = auto_encoder_circuit(num_latent, num_trash)

    # Wrap into a SamplerQNN
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["AutoencoderQML"]
