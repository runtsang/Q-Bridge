"""samplerqnn__gen059.py – Quantum sampler network with a variational 3‑qubit circuit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

__all__ = ["SamplerQNN"]


class SamplerQNN:
    """
    Variational quantum sampler that maps 2‑dimensional classical inputs to a 2‑class probability distribution.
    The circuit uses 3 qubits, entanglement layers, and a shared set of rotation parameters.
    """

    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        input_params: ParameterVector | None = None,
        weight_params: ParameterVector | None = None,
        sampler: StatevectorSampler | None = None,
    ) -> None:
        # Build default circuit if none provided
        if circuit is None:
            circuit = self._build_default_circuit()
        if input_params is None:
            input_params = ParameterVector("input", 2)
        if weight_params is None:
            weight_params = ParameterVector("weight", 6)
        if sampler is None:
            sampler = StatevectorSampler()

        self.circuit = circuit
        self.input_params = input_params
        self.weight_params = weight_params
        self.sampler = sampler
        self.sampler_qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    @staticmethod
    def _build_default_circuit() -> QuantumCircuit:
        """Create a 3‑qubit entangling variational circuit."""
        qc = QuantumCircuit(3)
        # Input rotations
        qc.ry(ParameterVector("input", 2)[0], 0)
        qc.ry(ParameterVector("input", 2)[1], 1)
        # Entanglement layer
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Parameterised rotations
        for i, qubit in enumerate(range(3)):
            qc.ry(ParameterVector("weight", 6)[i], qubit)
        # Second entanglement
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Final rotations
        for i, qubit in enumerate(range(3)):
            qc.ry(ParameterVector("weight", 6)[i + 3], qubit)
        return qc

    def forward(self, inputs: np.ndarray | list[list[float]]) -> np.ndarray:
        """
        Compute the probability distribution for each input sample.

        Parameters
        ----------
        inputs: np.ndarray or list
            Shape (N, 2) or (N, 2) Python list. Converted to a numpy array.

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) containing the probability of each class.
        """
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.float64)
        probs = self.sampler_qnn(inputs)
        return probs

    def sample(self, inputs: np.ndarray | list[list[float]], n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the categorical distribution defined by the circuit.

        Parameters
        ----------
        inputs: np.ndarray or list
            Shape (N, 2) or (N, 2) Python list.
        n_samples: int
            Number of samples per input.

        Returns
        -------
        np.ndarray
            Integer samples of shape (N, n_samples).
        """
        probs = self.forward(inputs)
        return np.random.choice(2, size=(probs.shape[0], n_samples), p=probs.T)

    @staticmethod
    def train_on_data(
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> "SamplerQNN":
        """
        Simple optimizer‑based training loop using the StatevectorSampler gradient backend.
        """
        from qiskit.algorithms.optimizers import COBYLA

        qnn = SamplerQNN()
        optimizer = COBYLA(maxiter=epochs)
        # Flatten initial parameters
        params = np.append(
            np.zeros(2),  # input params not optimised
            np.zeros(6),  # weight params
        )

        def cost_func(p):
            # assign weight params
            qnn.sampler_qnn.set_weights(p[2:])
            probs = qnn.forward(X)
            # cross‑entropy loss
            log_probs = np.log(probs + 1e-12)
            loss = -np.mean(log_probs[np.arange(len(y)), y])
            return loss

        optimizer.optimize(num_vars=len(params), objective_function=cost_func, initial_point=params)
        return qnn
