import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.utils import algorithm_globals

class HybridAutoencoder:
    """
    Quantum variant of :class:`HybridAutoencoder` that uses a variational
    RealAmplitudes ansatz for encoding and a swap‑test for reconstruction.
    The class exposes the same public methods as its classical counterpart.
    """

    def __init__(self, cfg: dict, *, seed: int = 42) -> None:
        algorithm_globals.random_seed = seed
        self.cfg = cfg
        self.input_dim = cfg["input_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.num_qubits = cfg.get("num_qubits", self.latent_dim)
        self.sampler = Sampler()
        self._build_circuit()

    def _build_circuit(self) -> None:
        """
        Construct a circuit that encodes the input into a latent
        representation, performs a swap test against a fresh copy,
        and measures a single classical bit that indicates
        reconstruction fidelity.
        """
        qr = QuantumRegister(self.num_qubits + 1, "q")  # +1 for auxiliary
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Encoder
        encoder = RealAmplitudes(self.num_qubits, reps=3)
        self.circuit.compose(encoder, range(self.num_qubits), inplace=True)

        # Swap test
        aux = self.num_qubits
        self.circuit.h(aux)
        for i in range(self.num_qubits):
            self.circuit.cswap(aux, i, i)  # swap with itself (identity)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the circuit with the given classical input.  The input must
        have shape (batch, num_qubits) and be normalised to [-1, 1].
        """
        if inputs.shape[1]!= self.num_qubits:
            raise ValueError("Input dimensionality does not match circuit qubits")
        shots = 1024
        results = self.sampler.run(self.circuit, parameter_binds=[inputs], shots=shots)
        return results.measurement_counts

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """
        For the pure swap‑test architecture the decoder is identical to the
        encoder; we simply re‑run the circuit on the latent state.
        """
        return self.encode(latents)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.decode(inputs)

    def train(self, data: np.ndarray, *, epochs: int = 20, learning_rate: float = 1e-3
    ) -> List[float]:
        """
        Very simple training loop that optimises the parameters of the
        RealAmplitudes ansatz using COBYLA.  The loss is the mean
        squared error between the circuit output (interpreted as a
        probability) and the target (the original input).
        """
        optimizer = COBYLA()
        history: List[float] = []

        def objective(params: np.ndarray) -> float:
            self.circuit.set_parameters(params)
            result = self.sampler.run(self.circuit, shots=1024).measurement_counts
            probs = np.array([result.get("0", 0) / 1024 for _ in range(len(data))])
            loss = np.mean((probs - data) ** 2)
            return loss

        for _ in range(epochs):
            params, loss = optimizer.optimize(
                num_vars=self.circuit.num_parameters, objective_function=objective
            )
            history.append(loss)
        return history

    def __repr__(self) -> str:  # pragma: no cover
        return f"<HybridAutoencoder quantum num_qubits={self.num_qubits}>"
