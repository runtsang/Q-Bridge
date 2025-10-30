"""Quantum implementation of a hybrid fully connected + quanvolution layer.

The class mirrors the classical counterpart but replaces the
convolutional front‑end with a parameterised quantum circuit that
encodes 2×2 image patches.  The output of each circuit is an
expectation value of Pauli‑Z, which is then fed into a classical
linear head.  The ``run`` method demonstrates how a list of
parameters can be interpreted as a quantum circuit and evaluated on
a backend.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable


class HybridFCLQuanvolution:
    """
    Quantum hybrid of a quanvolution filter and a fully connected head.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used to encode a 2×2 pixel patch.
    backend : qiskit.providers.BaseBackend, optional
        Backend to run the circuits on; defaults to the local
        ``qasm_simulator``.
    shots : int, default 100
        Number of shots for expectation estimation.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        backend=None,
        shots: int = 100,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Classical linear head weights (random for demo)
        self.linear_weights = np.random.randn(n_qubits * 14 * 14, 10)

    def _encode_patch(self, patch: np.ndarray) -> QuantumCircuit:
        """
        Encode a 2×2 patch into a circuit of size ``n_qubits``.

        Parameters
        ----------
        patch : np.ndarray
            Flattened pixel values, shape (n_qubits,).
        Returns
        -------
        QuantumCircuit
            Parameterised circuit ready for execution.
        """
        theta = Parameter("θ")
        qc = QuantumCircuit(self.n_qubits)
        # Simple encoding: Ry(θ) on each qubit, where θ is the pixel value
        for i in range(self.n_qubits):
            qc.ry(theta, i)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate a list of parameters as a quantum circuit.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters that would normally drive a quantum circuit.
        Returns
        -------
        np.ndarray
            Array of expectation values, one per theta.
        """
        expectations = []
        for theta in thetas:
            qc = QuantumCircuit(1)
            theta_param = Parameter("θ")
            qc.h(0)
            qc.ry(theta_param, 0)
            qc.measure_all()
            bound_qc = qc.bind_parameters({theta_param: theta})
            job = execute(
                bound_qc,
                backend=self.backend,
                shots=self.shots,
            )
            result = job.result()
            counts = result.get_counts(bound_qc)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        return np.array(expectations)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Full forward pass on a 28×28 single‑channel image.

        Parameters
        ----------
        image : np.ndarray
            Input image, shape (28, 28) and dtype float in [0, 1].
        Returns
        -------
        np.ndarray
            Log‑softmax logits of shape (10,).
        """
        # Extract 2×2 patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = image[r : r + 2, c : c + 2].flatten()
                patches.append(patch)

        # Evaluate each patch with the quantum encoder
        q_expectations = []
        for patch in patches:
            qc = self._encode_patch(patch)
            bound_qc = qc.bind_parameters({Parameter("θ"): np.mean(patch)})
            job = execute(
                bound_qc,
                backend=self.backend,
                shots=self.shots,
            )
            result = job.result()
            counts = result.get_counts(bound_qc)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            expectation = np.sum(states * probs)
            q_expectations.append(expectation)

        features = np.array(q_expectations).reshape(1, -1)
        logits = features @ self.linear_weights  # shape (1, 10)
        exp_logits = np.exp(logits)
        log_softmax = logits - np.log(np.sum(exp_logits, axis=1, keepdims=True))
        return log_softmax.squeeze()
