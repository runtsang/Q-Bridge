"""Enhanced quantum classifier mirroring the classical helper interface.

Features added compared to the seed:
- Parameter‑shift gradient computation
- Optional Aer noise model
- Configurable optimizer (gradient descent on parameters)
- Early stopping on validation loss
- Batch processing of data via statevector simulator
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional
import numpy as np

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer.noise import NoiseModel


class QuantumClassifierModel:
    """Quantum classifier with a classical‑style API."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str = "aer_simulator_statevector",
        shots: int = 1024,
        noise_model: Optional[NoiseModel] = None,
        lr: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits in the circuit.
        depth : int
            Depth of the variational ansatz.
        backend : str
            Qiskit Aer backend name (statevector or qasm).
        shots : int
            Number of shots for expectation estimation.
        noise_model : NoiseModel | None
            Optional noise model to attach to the backend.
        lr : float
            Learning rate for gradient descent.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend_name = backend
        self.shots = shots
        self.noise_model = noise_model
        self.lr = lr

        self.encoding_params = ParameterVector("x", self.num_qubits)
        self.weight_params = ParameterVector("theta", self.num_qubits * self.depth)
        self.circuit, self.observables = self._build_circuit()

        # initialise trainable parameters uniformly in [-π, π]
        self.weight_vals = np.random.uniform(-np.pi, np.pi, len(self.weight_params))

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[SparsePauliOp]]:
        """Construct a layered ansatz with data encoding and variational parameters."""
        qc = QuantumCircuit(self.num_qubits)
        for i, param in enumerate(self.encoding_params):
            qc.rx(param, i)

        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qc.ry(self.weight_params[idx], i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)

        # Two observables for binary classification
        if self.num_qubits >= 2:
            observables = [
                SparsePauliOp("Z" + "I" * (self.num_qubits - 1)),
                SparsePauliOp("I" + "Z" * (self.num_qubits - 1)),
            ]
        else:
            observables = [SparsePauliOp("Z"), SparsePauliOp("Z")]
        return qc, observables

    def _expectation(self, data: np.ndarray, weight_vals: np.ndarray) -> np.ndarray:
        """Compute expectation values of the observables for a single data point."""
        param_dict = {p: val for p, val in zip(self.encoding_params, data)}
        for i, p in enumerate(self.weight_params):
            param_dict[p] = weight_vals[i]

        bound_qc = self.circuit.bind_parameters(param_dict)

        if self.backend_name == "aer_simulator_statevector":
            backend = Aer.get_backend("statevector_simulator")
            job = execute(bound_qc, backend)
            sv = Statevector(job.result().get_statevector(bound_qc))
            exps = np.array([sv.expectation_value(obs).real for obs in self.observables])
        else:
            backend = Aer.get_backend(self.backend_name)
            job = execute(bound_qc, backend, shots=self.shots, noise_model=self.noise_model)
            result = job.result()
            counts = result.get_counts(bound_qc)
            exps = np.zeros(len(self.observables))
            for i, obs in enumerate(self.observables):
                z_counts = 0
                for bitstring, cnt in counts.items():
                    bit = int(bitstring[::-1][i])  # Qiskit returns little‑endian
                    z_counts += cnt * (1 if bit == 0 else -1)
                exps[i] = z_counts / self.shots
        return exps

    def _gradient(self, data: np.ndarray, label: int) -> np.ndarray:
        """Compute parameter‑shift gradient for a single data point."""
        grad = np.zeros_like(self.weight_vals)
        shift = np.pi / 2
        for idx in range(len(self.weight_vals)):
            shift_vec = self.weight_vals.copy()
            shift_vec[idx] += shift
            exp_plus = self._expectation(data, shift_vec)[label]

            shift_vec[idx] -= 2 * shift
            exp_minus = self._expectation(data, shift_vec)[label]

            grad[idx] = 0.5 * (exp_plus - exp_minus)
        return grad

    def train(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        epochs: int = 20,
        patience: int = 3,
        verbose: bool = True,
    ) -> None:
        """Gradient‑descent training with early stopping."""
        best_val = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.weight_vals = np.clip(self.weight_vals, -np.pi, np.pi)
            for x_batch, y_batch in train_loader:
                batch_grad = np.zeros_like(self.weight_vals)
                for x, y in zip(x_batch, y_batch):
                    batch_grad += self._gradient(x, y)
                batch_grad /= len(x_batch)
                self.weight_vals -= self.lr * batch_grad

            val_loss = self.evaluate(val_loader, return_loss=True)
            if verbose:
                print(f"Epoch {epoch+1:02d} | Val loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_params = self.weight_vals.copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print("Early stopping")
                    break
        self.weight_vals = best_params

    def evaluate(
        self,
        loader: Iterable,
        return_loss: bool = False,
    ) -> float:
        """Compute loss or accuracy over a dataset."""
        total = 0
        correct = 0
        loss_sum = 0.0
        for x_batch, y_batch in loader:
            for x, y in zip(x_batch, y_batch):
                exps = self._expectation(x, self.weight_vals)
                logits = exps
                probs = np.exp(logits) / np.sum(np.exp(logits))
                loss = -np.log(probs[y] + 1e-10)
                loss_sum += loss
                pred = np.argmax(probs)
                correct += int(pred == y)
                total += 1
        if return_loss:
            return loss_sum / total
        return correct / total

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of inputs."""
        probs = []
        for sample in x:
            exps = self._expectation(sample, self.weight_vals)
            logits = exps
            probs.append(np.exp(logits) / np.sum(np.exp(logits)))
        return np.array(probs)

    def get_encoding(self) -> Tuple[List[int], List[int], List[int]]:
        """Return metadata similar to the classical version."""
        encoding = list(range(self.num_qubits))
        weight_sizes = [len(self.weight_params)]
        observables = [0, 1]  # placeholder
        return encoding, weight_sizes, observables

    def __repr__(self) -> str:
        return f"<QuantumClassifierModel depth={self.depth} qubits={self.num_qubits}>"

__all__ = ["QuantumClassifierModel"]
