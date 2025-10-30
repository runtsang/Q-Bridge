"""Quantum convolutional filter with hybrid variational classifier.

The interface mirrors the classical `Conv` factory:
* `Conv(kernel_size, threshold, depth, classifier_depth)` returns an
  object that can be called on a 2‑D array and optionally
  classified.
* The quantum filter encodes a kernel‑sized patch into a block of
  qubits, applies a depth‑parameterized variational ansatz,
  and measures each qubit in the Z basis.
* The measurement outcomes are combined into a feature vector that
  is fed into a small classical linear head (implemented with PyTorch).
* The design allows the quantum part to be swapped with a classical
  implementation without changing downstream code.

The module relies only on Qiskit and PyTorch, keeping the quantum
contributions explicit while still enabling supervised learning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp

# --------------------------------------------------------------------------- #
# Helper: build classical classifier head
# --------------------------------------------------------------------------- #
def _build_quantum_classifier_head(in_features: int, depth: int) -> nn.Sequential:
    """Simple feed‑forward head used after the quantum feature extraction.

    Parameters
    ----------
    in_features : int
        Dimensionality of the quantum feature vector (number of qubits).
    depth : int
        Depth of the hidden layers.

    Returns
    -------
    nn.Sequential
        A PyTorch module that maps the quantum features to logits.
    """
    layers: List[nn.Module] = []
    in_dim = in_features
    hidden_size = in_features  # keep size constant for simplicity
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, 2))  # binary classification
    return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
# Core quantum convolutional filter
# --------------------------------------------------------------------------- #
class QuanvFilter:
    """Quantum convolutional filter with optional hybrid classifier."""

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 depth: int = 1,
                 classifier_depth: int | None = None,
                 shots: int = 1024,
                 backend_name: str = "qasm_simulator") -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.depth = depth
        self.shots = shots
        self.backend = Aer.get_backend(backend_name)

        # Parameter vectors for data encoding and variational weights
        self.x = ParameterVector("x", self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits * self.depth)

        # Build the ansatz
        self.circuit = QuantumCircuit(self.n_qubits)
        for q, param in zip(range(self.n_qubits), self.x):
            self.circuit.rx(param, q)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                self.circuit.ry(self.theta[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                self.circuit.cz(q, q + 1)

        # Measurement of each qubit in Z basis
        self.circuit.measure_all()

        # Observables for feature extraction
        self.observables: List[SparsePauliOp] = [
            SparsePauliOp(Pauli("Z" if i == j else "I" for i in range(self.n_qubits)).string)
            for j in range(self.n_qubits)
        ]

        # Optional classical classifier head
        self.classifier: nn.Module | None = None
        if classifier_depth is not None:
            self.classifier = _build_quantum_classifier_head(
                in_features=self.n_qubits,
                depth=classifier_depth
            )

    # ----------------------------------------------------------------------- #
    # Run the circuit and return a feature vector
    # ----------------------------------------------------------------------- #
    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute the quantum circuit on a single kernel patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) containing integer
            pixel values.

        Returns
        -------
        np.ndarray
            Expectation values of the Z observables for each qubit, shape
            (n_qubits,).
        """
        # Flatten and map data > threshold to π, else 0
        flattened = data.reshape(self.n_qubits)
        param_binds = {self.x[i]: (np.pi if val > self.threshold else 0)
                       for i, val in enumerate(flattened)}

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute expectation values for each qubit
        exp_vals = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            prob = cnt / self.shots
            # Convert bitstring to integer list
            bits = np.array([int(b) for b in bitstring[::-1]])  # reversed ordering
            # Z expectation: (+1 for 0, -1 for 1)
            exp_vals += (1 - 2 * bits) * prob

        return exp_vals

    # ----------------------------------------------------------------------- #
    # Optional classification
    # ----------------------------------------------------------------------- #
    def classify(self, data: np.ndarray) -> torch.Tensor:
        """Classify a single kernel patch using the hybrid head.

        Raises
        ------
        RuntimeError
            If the classifier head is not configured.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier head not configured.")
        features = self.run(data)
        torch_features = torch.tensor(features, dtype=torch.float32)
        return self.classifier(torch_features)

# --------------------------------------------------------------------------- #
# Public factory – identical signature to the classical one
# --------------------------------------------------------------------------- #
def Conv(kernel_size: int = 2,
         threshold: float = 0.0,
         depth: int = 1,
         classifier_depth: int | None = None,
         shots: int = 1024,
         backend_name: str = "qasm_simulator") -> QuanvFilter:
    """Return a quantum convolutional filter with optional hybrid classifier.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the convolution kernel.
    threshold : float, optional
        Threshold for data encoding.
    depth : int, optional
        Depth of the variational ansatz.
    classifier_depth : int, optional
        Depth of the hybrid classical head.
    shots : int, optional
        Number of shots for the simulator.
    backend_name : str, optional
        Qiskit backend to use.

    Returns
    -------
    QuanvFilter
        An instance ready to be called on a 2‑D patch.
    """
    return QuanvFilter(kernel_size=kernel_size,
                       threshold=threshold,
                       depth=depth,
                       classifier_depth=classifier_depth,
                       shots=shots,
                       backend_name=backend_name)

__all__ = ["Conv", "QuanvFilter"]
