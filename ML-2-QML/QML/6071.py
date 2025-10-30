import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

class Autoencoder__gen491:
    """Quantum auto‑encoder using a variational circuit and hybrid loss.

    The circuit encodes input data via RY gates, applies an encoder ansatz,
    entangles the latent qubits, then a decoder ansatz, and finally measures
    all qubits.  The loss is the mean‑squared error between the measurement
    probabilities and the original input.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        reps: int = 2,
        shots: int = 1024,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.shots = shots

        self.backend = AerSimulator()
        self.q_inst = QuantumInstance(self.backend, shots=self.shots)

        # Build circuit and separate trainable parameters
        self.circuit = self._build_circuit()
        self.param_dict = {p: 0.0 for p in self.circuit.parameters}
        # Identify trainable parameters (exclude input parameters)
        self.input_params = {p for p in self.circuit.parameters if p.name.startswith("x_")}
        self.trainable_params = [p for p in self.circuit.parameters if p not in self.input_params]
        self.param_values = np.zeros(len(self.trainable_params))

    def _build_circuit(self):
        qr = QuantumRegister(self.input_dim, "q")
        circuit = QuantumCircuit(qr)

        # Input encoding: RY gates with parameters x_i
        for i in range(self.input_dim):
            param = Parameter(f"x_{i}")
            circuit.ry(param, qr[i])

        # Encoder ansatz
        encoder = RealAmplitudes(self.input_dim, reps=self.reps)
        circuit.compose(encoder, qubits=qr, inplace=True)

        # Latent entanglement
        for i in range(self.latent_dim):
            circuit.cx(qr[i], qr[(i + 1) % self.input_dim])

        # Decoder ansatz
        decoder = RealAmplitudes(self.input_dim, reps=self.reps)
        circuit.compose(decoder, qubits=qr, inplace=True)

        # Measurement
        circuit.measure_all()
        return circuit

    def _evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Run the circuit for a single sample and return a probability vector."""
        # Build parameter mapping
        param_map = {f"x_{i}": float(sample[i]) for i in range(self.input_dim)}
        param_map.update({p: val for p, val in zip(self.trainable_params, self.param_values)})

        # Execute circuit
        result = self.q_inst.execute(self.circuit.assign_parameters(param_map, inplace=False))
        counts = result.get_counts()
        probs = np.zeros(self.input_dim)

        # Convert counts to probability distribution
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            # bitstring is a string of bits, e.g., '0101'
            for idx, bit in enumerate(reversed(bitstring)):
                probs[idx] += cnt * int(bit)
        probs /= total
        return probs

    def _loss_function(self, data: np.ndarray, params: np.ndarray) -> float:
        """Compute MSE loss over the dataset for a given set of parameters."""
        self.param_values = params
        loss = 0.0
        for sample in data:
            probs = self._evaluate(sample)
            loss += np.mean((probs - sample) ** 2)
        return loss / len(data)

    def train(self, data: np.ndarray, *, epochs: int = 10, verbose: bool = False):
        """Train the variational auto‑encoder using a global optimizer."""
        # Initial parameters
        params = np.zeros(len(self.trainable_params))

        for epoch in range(epochs):
            # Optimize
            res = minimize(
                lambda p: self._loss_function(data, p),
                params,
                method="COBYLA",
                options={"maxiter": 10},
            )
            params = res.x
            loss = self._loss_function(data, params)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

        self.param_values = params

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return reconstructed outputs for the given dataset."""
        reconstructions = []
        for sample in data:
            probs = self._evaluate(sample)
            reconstructions.append(probs)
        return np.array(reconstructions)
