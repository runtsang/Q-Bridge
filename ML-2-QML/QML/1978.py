"""Variational quantum regressor with layered ansatz and gradient‑based training."""

from __future__ import annotations
from typing import Iterable, Tuple, Callable

import numpy as np
import torch
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNNGen:
    """Quantum neural network with a repeated layer of single‑qubit rotations
    and CNOT entanglement. The network is trained using a state‑vector
    estimator and analytic gradients.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz. Must be at least 1.
    layers : int
        Number of identical parameterized layers.
    backend : str, optional
        Backend name for the estimator. Defaults to ``'statevector'``.
    """
    def __init__(
        self,
        num_qubits: int = 2,
        layers: int = 2,
        backend: str | None = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.layers = layers
        self.backend = backend or "statevector"

        # Build the ansatz
        self.input_params, self.weight_params = self._build_ansatz()
        self.circuit = self._assemble_circuit()
        self.observable = self._observable()
        self.estimator = StatevectorEstimator(backend=self.backend)
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def _build_ansatz(self) -> Tuple[list[Parameter], list[Parameter]]:
        """Create Parameter lists for inputs and trainable weights."""
        input_params = [Parameter(f"inp_{i}") for i in range(self.num_qubits)]
        weight_params = [Parameter(f"w_{l}_{i}") for l in range(self.layers) for i in range(self.num_qubits * 2)]
        return input_params, weight_params

    def _layer(self, qc: QuantumCircuit, layer_idx: int) -> None:
        """Append a single layer of H -> RY -> RX rotations followed by a CNOT ring."""
        for q in range(self.num_qubits):
            qc.h(q)
            qc.ry(self.weight_params[layer_idx * self.num_qubits * 2 + q], q)
            qc.rx(self.weight_params[layer_idx * self.num_qubits * 2 + self.num_qubits + q], q)
        # Entanglement cycle
        for q in range(self.num_qubits):
            qc.cx(q, (q + 1) % self.num_qubits)

    def _assemble_circuit(self) -> QuantumCircuit:
        """Build the full parameterized circuit."""
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding: rotation about Y by the input parameters
        for q in range(self.num_qubits):
            qc.ry(self.input_params[q], q)
        # Parameterized layers
        for l in range(self.layers):
            self._layer(qc, l)
        return qc

    def _observable(self) -> SparsePauliOp:
        """Expectation value of Pauli‑Z on the first qubit."""
        return SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _default_optimizer(self, lr: float = 0.01) -> torch.optim.Optimizer:
        """Return a simple Adam optimizer for the QNN parameters."""
        return torch.optim.Adam(self.qnn.parameters(), lr=lr)

    def train(
        self,
        data: Iterable[Tuple[np.ndarray, float]],
        epochs: int = 20,
        lr: float = 0.01,
        loss_fn: Callable[[float, float], float] | None = None,
    ) -> list[float]:
        """Train the QNN on the provided dataset.

        Parameters
        ----------
        data
            Iterable of ``(input_vector, target)`` pairs. ``input_vector`` must
            have length ``num_qubits``.
        epochs
            Number of training epochs.
        lr
            Learning rate for the optimizer.
        loss_fn
            Optional loss function; defaults to mean squared error.
        Returns
        -------
        losses
            List of epoch‑wise training losses.
        """
        if loss_fn is None:
            loss_fn = lambda pred, tgt: (pred - tgt) ** 2

        optimizer = self._default_optimizer(lr)
        losses: list[float] = []

        for _ in range(epochs):
            epoch_losses: list[float] = []
            for inp_vec, tgt in data:
                inp = torch.tensor(inp_vec, dtype=torch.float32)
                tgt_tensor = torch.tensor(tgt, dtype=torch.float32)
                optimizer.zero_grad()
                pred = self.qnn(inp)
                loss = loss_fn(pred, tgt_tensor)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(sum(epoch_losses) / len(epoch_losses))
        return losses

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of inputs.

        Parameters
        ----------
        inputs
            2‑D array of shape (batch_size, num_qubits).
        Returns
        -------
        preds
            Array of shape (batch_size,) with the predicted expectation values.
        """
        preds = []
        for inp_vec in inputs:
            inp = torch.tensor(inp_vec, dtype=torch.float32)
            pred = self.qnn(inp).item()
            preds.append(pred)
        return np.array(preds)
