import numpy as np
import qiskit
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

__all__ = ["QuantumAutoencoder"]

class QuantumAutoencoder:
    """
    Quantum autoencoder that maps an input vector to a latent representation
    using a parameterized RealAmplitudes circuit and a swap‑test style
    measurement.  The circuit is wrapped in a SamplerQNN to expose a
    differentiable forward interface.
    """
    def __init__(self,
                 num_qubits: int,
                 backend=None,
                 shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(num_qubits)
        self.input_params = [qiskit.circuit.Parameter(f"x{i}") for i in range(num_qubits)]

        self.ansatz = RealAmplitudes(num_qubits, reps=2)
        self.circuit.append(self.ansatz, range(num_qubits))

        self.circuit.measure_all()

        self.weight_params = self.ansatz.parameters

        def interpret(x):
            return x

        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=interpret,
            output_shape=(num_qubits,),
            sampler=Sampler(),
        )
        init_weights = np.random.randn(len(self.weight_params))
        self.qnn.set_weights(init_weights)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the quantum autoencoder.
        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch_size, num_qubits).
        Returns
        -------
        np.ndarray
            Shape (batch_size, num_qubits) containing expectation values.
        """
        return self.qnn(inputs)

    def get_weights(self) -> np.ndarray:
        return self.qnn.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        self.qnn.set_weights(weights)

    def train(self,
              data: np.ndarray,
              target: np.ndarray,
              epochs: int = 50) -> None:
        """
        Train the quantum encoder using COBYLA to minimize MSE between
        the circuit output and the target vector.
        """
        weight_shape = len(self.get_weights())

        def objective(weights: np.ndarray) -> float:
            self.set_weights(weights)
            outputs = self.run(data)
            loss = np.mean((outputs - target) ** 2)
            return loss

        optimizer = COBYLA(maxiter=2000, disp=False)
        x0 = self.get_weights()
        optimizer.optimize(weight_shape, objective, x0)
        self.set_weights(optimizer.x)
