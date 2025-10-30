"""Quantum classifier class with advanced variational ansatz and training.

Highlights
----------
* Supports multiple ansatz styles (simple, entangled, custom).
* Can run on local Aer simulators or remote IBMQ backends.
* Provides a gradient‑based training loop using the parameter‑shift rule.
* Offers a flexible interface mirroring the classical counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq import IBMQ
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, ExpectationFactory
from qiskit.opflow.gradients import ParameterShift

# Import the original circuit builder
from.QuantumClassifierModel import build_classifier_circuit


class QuantumClassifierModel:
    """Variational quantum classifier with training utilities.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Depth of the variational ansatz.
    ansatz : str, default "simple"
        Type of ansatz: "simple", "entangled", or a callable returning a circuit.
    backend : str | qiskit.providers.Provider, default "aer_simulator"
        Backend name or instance.  Supports local Aer or IBMQ providers.
    shots : int, default 1024
        Number of shots for expectation estimation.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        ansatz: str | Callable[[QuantumCircuit, List[ParameterVector], List[ParameterVector]], None] = "simple",
        backend: str | object = "aer_simulator",
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots

        # Build base circuit and metadata
        circuit, encoding, weights, observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.base_circuit = circuit
        self.encoding = encoding
        self.weights = weights
        self.observables = observables

        # Replace ansatz if requested
        if isinstance(ansatz, str) and ansatz!= "simple":
            self._apply_ansatz(ansatz)
        elif callable(ansatz):
            ansatz(self.base_circuit, self.encoding, self.weights)

        # Backend handling
        if isinstance(backend, str):
            if backend == "aer_simulator":
                self.backend = AerSimulator()
            else:
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
                self.backend = provider.get_backend(backend)
        else:
            self.backend = backend

        # Pre‑compile for speed
        self.compiled_circuit = transpile(self.base_circuit, self.backend)

        # Gradient evaluator
        self.gradient = ParameterShift()

    # ------------------------------------------------------------------
    # Ansatz utilities
    # ------------------------------------------------------------------
    def _apply_ansatz(self, ansatz_type: str) -> None:
        """Replace the default ansatz with a richer entangling pattern."""
        circuit = self.base_circuit
        if ansatz_type == "entangled":
            # Add a layer of CNOTs in a ring topology
            for _ in range(self.depth):
                for q in range(self.num_qubits):
                    circuit.cx(q, (q + 1) % self.num_qubits)
        else:
            raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def circuit_for_params(self, params: np.ndarray) -> QuantumCircuit:
        """Return a circuit with all parameters bound to ``params``."""
        if len(params)!= len(self.encoding) + len(self.weights):
            raise ValueError("Parameter vector length mismatch.")
        bound_circuit = self.base_circuit.bind_parameters(
            {p: v for p, v in zip(self.encoding + self.weights, params)}
        )
        return transpile(bound_circuit, self.backend)

    # ------------------------------------------------------------------
    # Expectation evaluation
    # ------------------------------------------------------------------
    def expectation_values(self, params: np.ndarray) -> np.ndarray:
        """Return expectation values of the observables for given parameters."""
        circ = self.circuit_for_params(params)
        job = self.backend.run(circ, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circ)
        exp_vals = np.zeros(len(self.observables))
        for i, op in enumerate(self.observables):
            exp_vals[i] = self._expectation_from_counts(counts, op)
        return exp_vals

    def _expectation_from_counts(self, counts: dict, op: SparsePauliOp) -> float:
        """Compute expectation value of a Pauli operator from measurement counts."""
        # Convert Pauli to bitstring basis
        pauli_str = op.to_label()
        exp = 0.0
        for bitstring, n in counts.items():
            parity = 1
            for qubit, pauli in enumerate(reversed(pauli_str)):
                if pauli == "Z" and bitstring[qubit] == "1":
                    parity = -parity
            exp += parity * n
        return exp / sum(counts.values())

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return class predictions for a batch of data samples.

        Each sample is encoded into the circuit parameters via the
        encoding vector ``self.encoding``.
        """
        predictions = []
        for sample in data:
            params = np.concatenate([sample, np.zeros(len(self.weights))])
            exp_vals = self.expectation_values(params)
            predictions.append(np.argmax(exp_vals))
        return np.array(predictions)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train(
        self,
        data_loader: Iterable,
        lr: float = 0.01,
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """Gradient‑based training loop using the parameter‑shift rule."""
        # Initialize parameters
        params = np.random.randn(len(self.encoding) + len(self.weights))
        optimizer = lambda p, g, lr: p - lr * g  # simple SGD

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in data_loader:
                batch_x = np.asarray(batch_x)
                batch_y = np.asarray(batch_y)
                grads = np.zeros_like(params)

                # Compute loss and gradients
                for i, sample in enumerate(batch_x):
                    # Forward
                    params_sample = np.concatenate([sample, params[len(self.encoding) :]])
                    exp_vals = self.expectation_values(params_sample)
                    loss = -np.log(exp_vals[batch_y[i]] + 1e-10)
                    epoch_loss += loss

                    # Backward via parameter shift
                    for j in range(len(params)):
                        shift = np.zeros_like(params)
                        shift[j] = np.pi / 2
                        f_plus = self.expectation_values(params_sample + shift)[batch_y[i]]
                        f_minus = self.expectation_values(params_sample - shift)[batch_y[i]]
                        grads[j] += (f_plus - f_minus) / 2

                grads /= len(batch_x)
                params = optimizer(params, grads, lr)

            if verbose:
                avg_loss = epoch_loss / len(data_loader.dataset)
                print(f"Epoch {epoch+1}/{epochs} – loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, data_loader: Iterable) -> Tuple[float, float]:
        """Return (accuracy, loss) on the validation set."""
        correct, total, total_loss = 0, 0, 0.0
        for batch_x, batch_y in data_loader:
            preds = self.predict(batch_x)
            correct += np.sum(preds == batch_y)
            total += len(batch_x)
            # Simple cross‑entropy loss in log‑space
            for i, sample in enumerate(batch_x):
                params = np.concatenate([sample, np.zeros(len(self.weights))])
                exp_vals = self.expectation_values(params)
                loss = -np.log(exp_vals[batch_y[i]] + 1e-10)
                total_loss += loss
        accuracy = correct / total
        return accuracy, total_loss / total

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        """Return total number of variational parameters."""
        return len(self.encoding) + len(self.weights)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_qubits={self.num_qubits}, "
            f"depth={self.depth}, params={self.num_parameters()}, "
            f"backend={self.backend.name if hasattr(self.backend, 'name') else self.backend})"
        )


__all__ = ["QuantumClassifierModel"]
