"""Hybrid quantum regressor inspired by EstimatorQNN and Autoencoder.

This module implements a `HybridEstimator` that uses a variational
circuit with a RealAmplitudes ansatz and a swap‑test‑style
measurement to produce a scalar regression output.  The circuit
encodes input features via Ry rotations, applies a variational
ansatz on latent qubits, and measures the last qubit.  The
`EstimatorQNN` wrapper turns the circuit into a differentiable
neural network that can be trained with gradient‑based optimisers.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np

class HybridEstimator:
    """
    Quantum regression model that mirrors the classical HybridEstimator.
    Parameters
    ----------
    num_features : int
        Number of input features to encode.
    latent_dim : int, default 4
        Size of the latent sub‑space (number of variational qubits).
    reps : int, default 3
        Repetitions of the RealAmplitudes ansatz.
    """

    def __init__(self, num_features: int, *, latent_dim: int = 4, reps: int = 3) -> None:
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.reps = reps
        self.circuit = self._build_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=SparsePauliOp.from_list([("Z" * self.circuit.num_qubits, 1)]),
            input_params=[self.circuit.parameters[i] for i in range(num_features)],
            weight_params=[self.circuit.parameters[i] for i in range(num_features, self.circuit.num_qubits)],
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs the variational circuit used for regression."""
        qr = QuantumRegister(self.num_features + self.latent_dim, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode classical features
        for i in range(self.num_features):
            qc.ry(Parameter(f"input_{i}"), i)

        # Variational ansatz on latent qubits
        ansatz = RealAmplitudes(num_qubits=self.latent_dim, reps=self.reps)
        qc.compose(ansatz, range(self.num_features, self.num_features + self.latent_dim), inplace=True)

        # Measure the last latent qubit as the regression output
        qc.measure(self.num_features + self.latent_dim - 1, cr[0])
        return qc

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Return regression predictions for a batch of inputs.
        Parameters
        ----------
        inputs : np.ndarray of shape (batch, num_features)
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        preds = []
        for sample in inputs:
            bound = {self.circuit.parameters[i]: sample[i] for i in range(self.num_features)}
            expectation = self.qnn.predict(bound)[0]
            preds.append(expectation)
        return np.array(preds)

__all__ = ["HybridEstimator"]
