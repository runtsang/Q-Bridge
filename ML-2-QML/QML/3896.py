import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class AutoencoderQEMerge:
    """Quantum decoder that maps a latent vector to a reconstruction vector."""
    def __init__(self, config):
        """
        Parameters
        ----------
        config : AutoencoderQEMergeConfig
            Must provide ``input_dim`` and ``latent_dim`` attributes.
        """
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        # Number of auxiliary qubits used in the swap‑test pattern
        self.num_trash = max(1, self.latent_dim // 2)
        self.circuit = self._build_circuit()
        # Sampler for state‑vector simulation
        self.sampler = StatevectorSampler()
        # The SamplerQNN interprets the state‑vector to a reconstruction vector
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=list(range(self.latent_dim)),
                              weight_params=self.circuit.parameters,
                              interpret=self._interpret,
                              output_shape=(self.input_dim,),
                              sampler=self.sampler)

    def _interpret(self, statevector: np.ndarray) -> np.ndarray:
        """
        Convert a state‑vector into a reconstruction vector.
        Each entry is the probability of measuring a 1 on the corresponding input qubit.
        """
        probs = []
        for qubit in range(self.input_dim):
            mask = 1 << qubit
            prob = 0.0
            for idx, amp in enumerate(statevector):
                if idx & mask:
                    prob += abs(amp) ** 2
            probs.append(prob)
        return np.array(probs)

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the variational circuit used in the decoder."""
        latent_dim = self.latent_dim
        num_trash  = self.num_trash
        input_dim  = self.input_dim

        # Total qubits = input qubits (reconstruction) + latent + trash + ancilla
        total_qubits = input_dim + latent_dim + 2 * num_trash + 1
        qr = QuantumRegister(total_qubits, name="q")
        cr = ClassicalRegister(total_qubits, name="c")
        circuit = QuantumCircuit(qr, cr)

        # 1. Feature map: encode the latent vector into the first ``latent_dim`` qubits
        feature_map = RawFeatureVector(latent_dim, reps=1)
        circuit.compose(feature_map, inplace=True)

        # 2. Variational ansatz on the latent + trash qubits
        ansatz = RealAmplitudes(latent_dim + num_trash, reps=5)
        circuit.compose(ansatz, range(0, latent_dim + num_trash), inplace=True)

        # 3. Domain‑wall pattern on the first set of trash qubits
        for i in range(num_trash):
            circuit.x(latent_dim + i)

        # 4. Swap‑test with an ancilla qubit
        ancilla = latent_dim + 2 * num_trash
        circuit.h(ancilla)
        for i in range(num_trash):
            circuit.cswap(ancilla, latent_dim + i, latent_dim + num_trash + i)
        circuit.h(ancilla)
        circuit.measure(ancilla, cr[ancilla])

        # 5. Measure the ``input_dim`` qubits that will act as the reconstruction output
        for i in range(input_dim):
            circuit.measure(i, cr[i])

        return circuit

    def decode(self, latent_np: np.ndarray) -> np.ndarray:
        """
        Decode a batch of latent vectors into a reconstruction batch.

        Parameters
        ----------
        latent_np : np.ndarray
            Shape ``(batch_size, latent_dim)``.  The array must be of type ``float``.

        Returns
        -------
        recon : np.ndarray
            Shape ``(batch_size, input_dim)``.  Each entry is a probability between 0 and 1.
        """
        return self.qnn.forward(latent_np)

__all__ = ["AutoencoderQEMerge"]
