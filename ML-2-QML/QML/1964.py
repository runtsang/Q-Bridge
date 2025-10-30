"""Quantum sampler network with batched sampling and gradient training.

The implementation extends Qiskit's :class:`qiskit_machine_learning.neural_networks.SamplerQNN`
by adding convenience methods for:

* batched sampling using Aer simulators
* automatic differentiation of expectation values
* a simple training loop that optimizes the circuit parameters
  with a user‑supplied loss function.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Iterable
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN as _QiskitSamplerQNN
from qiskit.primitives import Sampler as QiskitSampler
from scipy.optimize import minimize


class SamplerQNN:
    """
    A wrapper around Qiskit's :class:`qiskit_machine_learning.neural_networks.SamplerQNN`
    that exposes batched sampling and a gradient‑based training helper.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterised quantum circuit.
    input_params : ParameterVector
        Parameters that encode the input to the sampler.
    weight_params : ParameterVector
        Trainable weight parameters.
    shots : int
        Number of shots per circuit evaluation.
    backend : str | AerSimulator
        Backend to use for simulation.  If a string is provided it is interpreted
        as an Aer simulator name (e.g. ``"qasm_simulator"``).
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        input_params: ParameterVector,
        weight_params: ParameterVector,
        shots: int = 1024,
        backend: str | AerSimulator = "qasm_simulator",
    ) -> None:
        if isinstance(backend, str):
            backend = AerSimulator(name=backend)
        self.backend = backend
        self.circuit = circuit
        self.input_params = input_params
        self.weight_params = weight_params
        self.shots = shots

        # Instantiate the underlying Qiskit SamplerQNN
        self.sampler_qnn = _QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=QiskitSampler(backend=self.backend),
        )

    # ------------------------------------------------------------------
    # Core functionality
    # ------------------------------------------------------------------
    def sample_batch(
        self,
        inputs: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Sample from the circuit for a batch of input vectors.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape ``(N, len(input_params))``.
        batch_size : int
            Number of inputs to evaluate per Aer run.

        Returns
        -------
        np.ndarray
            Samples of shape ``(N, shots)`` with values 0 or 1.
        """
        all_samples = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            bound_circuits = [
                self.circuit.bind_parameters(
                    dict(zip(self.input_params, inp))
                )
                for inp in batch
            ]
            transpiled = transpile(bound_circuits, backend=self.backend)
            results = self.backend.run(transpiled, shots=self.shots).result()
            for res in results.get_counts():
                # Convert counts dict to samples
                counts = res
                for outcome, freq in counts.items():
                    all_samples.extend([int(outcome) for _ in range(freq)])
        return np.array(all_samples).reshape(len(inputs), self.shots)

    def predict(
        self,
        inputs: np.ndarray,
        num_samples: int = 1,
    ) -> np.ndarray:
        """Return a probability estimate for the ``1`` outcome."""
        samples = self.sample_batch(inputs, batch_size=64)
        probs = samples.mean(axis=1, keepdims=True)
        return probs if num_samples == 1 else probs.repeat(num_samples, axis=0)

    def loss(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        loss_fn: Callable[[np.ndarray, np.ndarray], float] = np.mean,
    ) -> float:
        """Compute a simple loss between predictions and targets.

        The default loss is the mean squared error.
        """
        preds = self.predict(inputs)
        return loss_fn((preds - targets) ** 2)

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------
    def _loss_wrapper(self, weight_vals: np.ndarray, inputs: np.ndarray, targets: np.ndarray) -> float:
        # Bind new weight values
        bound_circuits = [
            self.circuit.bind_parameters(
                dict(zip(self.weight_params, weight_vals))
            )
            for _ in range(len(inputs))
        ]
        transpiled = transpile(bound_circuits, backend=self.backend)
        results = self.backend.run(transpiled, shots=self.shots).result()
        preds = []
        for res in results.get_counts():
            # compute probability of outcome '1'
            counts = res
            total = sum(counts.values())
            prob_one = sum(int(outcome) * freq for outcome, freq in counts.items()) / total
            preds.append(prob_one)
        preds = np.array(preds)
        return np.mean((preds - targets) ** 2)

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10,
        lr: float = 0.01,
        loss_fn: Callable[[np.ndarray, np.ndarray], float] = np.mean,
    ) -> dict:
        """Gradient‑based training loop for the weight parameters.

        Parameters
        ----------
        inputs : np.ndarray
            Training data of shape ``(N, len(input_params))``.
        targets : np.ndarray
            Target values in ``[0, 1]`` of shape ``(N,)``.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate for the optimizer.
        loss_fn : Callable
            Loss function that accepts predictions and targets.

        Returns
        -------
        dict
            Dictionary containing the optimized weights and loss history.
        """
        init_weights = np.array([float(v) for v in self.weight_params])
        loss_history = []

        def closure(x):
            loss_val = self._loss_wrapper(x, inputs, targets)
            loss_history.append(loss_val)
            return loss_val

        res = minimize(closure, init_weights, method="BFGS", options={"maxiter": epochs})
        return {"optimized_weights": res.x, "loss_history": loss_history}
