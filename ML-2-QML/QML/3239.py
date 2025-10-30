import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit.library import RealAmplitudes


class QuantumSelfAttentionAutoencoder:
    """
    Hybrid quantum implementation:
    1. A parameterised self‑attention circuit (rotations + C‑RX entanglement).
    2. A swap‑test autoencoder that learns a latent subspace.
    Both blocks are wrapped in SamplerQNNs for variational optimisation.
    """

    def __init__(self, n_qubits: int, latent_dim: int = 3, trash: int = 2,
                 backend=None):
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.trash = trash
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler()

        # Build circuits
        self.attention_circuit = self._build_attention_circuit()
        self.autoencoder_circuit = self._build_autoencoder_circuit()

        # Wrap in SamplerQNNs
        self.attention_qnn = SamplerQNN(
            circuit=self.attention_circuit,
            input_params=[],  # no external inputs – parameters are optimised
            weight_params=self.attention_circuit.parameters,
            interpret=lambda x: x,
            output_shape=(self.n_qubits,),
            sampler=self.sampler,
        )

        self.autoencoder_qnn = SamplerQNN(
            circuit=self.autoencoder_circuit,
            input_params=[],
            weight_params=self.autoencoder_circuit.parameters,
            interpret=lambda x: x,
            output_shape=(self.latent_dim,),
            sampler=self.sampler,
        )

    def _build_attention_circuit(self) -> QuantumCircuit:
        """Parameterised circuit that mimics classical self‑attention."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        rot = ParameterVector("rot", length=3 * self.n_qubits)
        ent = ParameterVector("ent", length=self.n_qubits - 1)

        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.measure(qr, cr)
        return qc

    def _build_autoencoder_circuit(self) -> QuantumCircuit:
        """Swap‑test autoencoder inspired by the seed implementation."""
        num_latent = self.latent_dim
        num_trash = self.trash
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Variational ansatz on latent + trash qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        qc.barrier()

        # Swap‑test auxiliary qubit
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def run(self, *args):
        """
        Execute the hybrid pipeline:
        1. Optimize attention parameters.
        2. Feed attention output into autoencoder parameters.
        Returns the raw sampler output of the autoencoder.
        """
        # In practice, optimisation loops would be orchestrated by a higher‑level
        # optimizer (e.g. COBYLA, SPSA).  Here we provide a synchronous run
        # that simply propagates parameters through both QNNs.

        # Attention step
        attention_counts = self.attention_qnn.run(*args)
        # Convert counts to expectation values (simple example)
        attention_out = np.array([int(k, 2) for k in attention_counts.keys()]) / sum(attention_counts.values())

        # Autoencoder step – in a real setting we would feed attention_out
        # as additional classical data or as part of the parameter vector.
        auto_counts = self.autoencoder_qnn.run(*args)
        auto_out = np.array([int(k, 2) for k in auto_counts.keys()]) / sum(auto_counts.values())
        return auto_out


__all__ = ["QuantumSelfAttentionAutoencoder"]
