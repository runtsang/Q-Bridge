import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, MeasureAll, RY, CX, RX
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class QuantumEncoder:
    """Variational encoder with swap‑test and domain‑wall."""
    def __init__(self, num_latent: int, num_trash: int, reps: int = 5) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps

    def circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        qc.barrier()
        # Domain wall: flip trash qubits
        for i in range(self.num_trash):
            qc.x(self.num_latent + i)
        # Swap test
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

class QuantumDecoder:
    """Sampler‑based decoder that maps a 2‑dim input to a 2‑dim output."""
    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        sampler = Sampler()
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        self.qnn = SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        # Map latent vector to 2‑dim input for the QNN
        inputs = np.stack([z[:, 0], z[:, 1]], axis=-1)
        return self.qnn(inputs)

class QuantumLSTMCell:
    """A minimal quantum LSTM cell using a small parameterized circuit."""
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.encoder = qiskit.circuit.library.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = [RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        self.measure = MeasureAll()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Placeholder: in a full implementation, x would be encoded into qubits,
        # the circuit would be executed, and measurement results returned.
        return x  # Identity for illustrative purposes

class UnifiedAutoencoder:
    """
    Quantum auto‑encoder with optional quantum LSTM for sequence latent processing.
    Shares the public API with the classical version.
    """
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        *,
        use_lstm: bool = False,
        n_qubits: int = 4
    ) -> None:
        self.encoder = QuantumEncoder(num_latent, num_trash)
        self.decoder = QuantumDecoder(num_latent)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = QuantumLSTMCell(n_qubits)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        # In practice, inputs would be bound to the encoder circuit.
        # Here we return a placeholder latent vector.
        return np.zeros((inputs.shape[0], self.encoder.num_latent))

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder(z)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        z = self.encode(inputs)
        if self.use_lstm:
            z = self.lstm(z)
        return self.decode(z)

__all__ = ["UnifiedAutoencoder"]
