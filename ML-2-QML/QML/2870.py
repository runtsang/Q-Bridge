import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler

class ConvAutoencoderHybrid:
    """
    Quantum implementation of a convolutional autoencoder.
    The input image is encoded into qubits using RX rotations (quantum convolution).
    A variational ansatz on latent qubits implements the latent layer, followed by a SWAP test
    for reconstruction. The output is the probability of measuring |1> on an auxiliary qubit.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 3,
                 shots: int = 1024):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.shots = shots
        self.backend = Sampler()
        self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.input_dim
        qc = QuantumCircuit(n_qubits, 1)

        # Quantum convolution: encode each input pixel with an RX rotation
        self.conv_params = ParameterVector('theta', n_qubits)
        for i, theta in enumerate(self.conv_params):
            qc.rx(theta, i)

        # Latent layer: variational ansatz on latent qubits
        latent_reg = QuantumRegister(self.latent_dim, "l")
        qc.add_register(latent_reg)
        self.latent_params = ParameterVector('phi', self.latent_dim)
        for i, phi in enumerate(self.latent_params):
            qc.ry(phi, i + n_qubits)

        # SWAP test for reconstruction
        aux = QuantumRegister(1, "aux")
        qc.add_register(aux)
        qc.h(aux[0])
        for i in range(self.latent_dim):
            qc.cswap(aux[0], i + n_qubits, i + n_qubits + self.latent_dim)
        qc.h(aux[0])
        qc.measure(aux[0], 0)

        self.circuit = qc

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit on the given data.
        Parameters
        ----------
        data : np.ndarray
            1D array with length equal to input_dim.
        Returns
        -------
        np.ndarray
            Probability of measuring |1> on the auxiliary qubit.
        """
        if data.size!= self.input_dim:
            raise ValueError("Input data must have length equal to input_dim")
        bind = {p: data[i] for i, p in enumerate(self.conv_params)}
        for p in self.latent_params:
            bind[p] = 0.0
        job = self.backend.run(self.circuit, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        prob_one = counts.get('1', 0) / self.shots
        return np.array([prob_one])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.run(data)
