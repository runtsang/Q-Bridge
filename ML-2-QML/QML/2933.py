"""Quantum hybrid autoencoder with swap‑test based latent representation and EstimatorQNN decoder."""

import numpy as np
import warnings
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Flip qubits a..b‑1 to introduce a domain wall."""
    for i in range(a, b):
        circuit.x(i)
    return circuit

def auto_encoder_circuit(num_latent: int, num_trash: int, reps: int = 5) -> QuantumCircuit:
    """Builds the swap‑test based encoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

class HybridAutoencoder:
    """Quantum autoencoder that encodes classical data into a latent qubit state and decodes via EstimatorQNN."""

    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        seed: int | None = 42,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = Sampler()
        self.estimator = Estimator()
        self.circuit = auto_encoder_circuit(num_latent, num_trash, reps)
        # Simple decoder circuit: single qubit H + Ry(theta)
        self.decoder = QuantumCircuit(1)
        self.decoder.h(0)
        self.decoder.ry(Parameter("theta"), 0)
        # Build EstimatorQNN for decoding
        self.qnn = EstimatorQNN(
            circuit=self.decoder,
            observables=[SparsePauliOp.from_list([("Y", 1)])],
            input_params=[Parameter("theta")],
            weight_params=[],
            estimator=self.estimator,
        )
        self.optimizer = COBYLA(maxiter=200)
        # Initialize parameters for the ansatz
        self.param_values = np.random.random(len(self.circuit.parameters))

    def encode(self, input_features: np.ndarray) -> np.ndarray:
        """Encode classical features into a quantum state and sample latent probabilities."""
        # Map input features to ansatz parameters
        param_dict = {p: f for p, f in zip(self.circuit.parameters, input_features)}
        qc = self.circuit.assign_parameters(param_dict, inplace=False)
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts(qc)
        probs = counts.get("0", 0) / 1024
        return np.array([probs])

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent probabilities back to classical space using EstimatorQNN."""
        theta = latent[0]
        output = self.qnn([theta])
        return np.array(output)

    def __call__(self, input_features: np.ndarray) -> np.ndarray:
        """Full autoencoder: encode then decode."""
        latent = self.encode(input_features)
        return self.decode(latent)

    def train(self, data: np.ndarray, *, epochs: int = 200, lr: float = 0.01) -> list[float]:
        """Train the quantum autoencoder by optimizing the ansatz parameters."""
        history: list[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x in data:
                # Forward
                latent = self.encode(x)
                out = self.decode(latent)
                loss = np.mean((out - x) ** 2)
                epoch_loss += loss
                # Gradient‑free optimizer step
                self.param_values = self.optimizer.minimize(
                    lambda p: self._loss(p, x),
                    self.param_values,
                )
            epoch_loss /= len(data)
            history.append(epoch_loss)
        return history

    def _loss(self, params: np.ndarray, x: np.ndarray) -> float:
        """Helper to compute loss for a single sample given parameters."""
        param_dict = {p: val for p, val in zip(self.circuit.parameters, params)}
        qc = self.circuit.assign_parameters(param_dict, inplace=False)
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts(qc)
        probs = counts.get("0", 0) / 1024
        latent = np.array([probs])
        out = self.decode(latent)
        return float(np.mean((out - x) ** 2))

__all__ = ["HybridAutoencoder"]
