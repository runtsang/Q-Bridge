"""Quantum hybrid sampler with integrated estimator.

The class builds a sampler circuit that encodes a 2‑D patch of data into
four qubits, applies a random layer, and samples the resulting state.
The sampled probabilities are then fed into a quantum estimator that
returns an expectation value of a Y‑observable.  The design mirrors the
classical architecture in :mod:`SamplerQNN` while providing a quantum
front‑end for the sampler and a quantum back‑end for regression.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

class HybridSamplerQNN:
    """
    Hybrid quantum sampler + estimator.

    The forward method accepts a batch of 28×28 images.  For each 2×2
    patch it encodes the pixel values into four qubits, runs the
    sampler circuit, and collects the probabilities.  These
    probabilities are then used as inputs to a quantum estimator
    that evaluates the expectation value of a Y‑observable across
    the sampled distribution.
    """

    def __init__(self, patch_size: int = 2, shots: int = 1024) -> None:
        self.patch_size = patch_size
        self.shots = shots

        # Sampler circuit: 4 qubits, 4 input parameters, 4 weight parameters
        self.input_params = ParameterVector("x", patch_size ** 2)
        self.weight_params = ParameterVector("w", patch_size ** 2)

        self.sampler_circuit = QuantumCircuit(patch_size ** 2)
        for i in range(patch_size ** 2):
            self.sampler_circuit.ry(self.input_params[i], i)
        self.sampler_circuit += random_circuit(patch_size ** 2, 2)
        self.sampler_circuit.measure_all()

        backend = qiskit.Aer.get_backend("qasm_simulator")
        sampler = StatevectorSampler(backend=backend, shots=self.shots)
        self.sampler_qnn = SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

        # Estimator circuit: 1 qubit, 2 parameters
        self.est_input = Parameter("input")
        self.est_weight = Parameter("weight")
        est_circuit = QuantumCircuit(1)
        est_circuit.h(0)
        est_circuit.ry(self.est_input, 0)
        est_circuit.rx(self.est_weight, 0)
        est_circuit.measure_all()

        observable = qiskit.quantum_info.SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator(backend=backend, shots=self.shots)
        self.estimator_qnn = EstimatorQNN(
            circuit=est_circuit,
            observables=observable,
            input_params=[self.est_input],
            weight_params=[self.est_weight],
            estimator=estimator,
        )

    def forward(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            images: NumPy array of shape (batch, 28, 28) with pixel values in [0, 1]

        Returns:
            probs: Sampled probabilities for each patch, shape (batch, num_patches, 4)
            expectation: Estimator expectation values, shape (batch,)
        """
        batch, h, w = images.shape
        assert h == w == 28, "Images must be 28×28"

        # Extract 2×2 patches
        patches = []
        for i in range(0, 28, self.patch_size):
            for j in range(0, 28, self.patch_size):
                patch = images[:, i : i + self.patch_size, j : j + self.patch_size]
                patches.append(patch.reshape(batch, -1))  # (batch, 4)

        # Run sampler on each patch
        probs_list = []
        for patch in patches:
            # Bind input parameters to pixel values
            bind = {self.input_params[i]: float(patch[:, i]) for i in range(patch.shape[1])}
            # For simplicity, use zero weights
            bind.update({self.weight_params[i]: 0.0 for i in range(patch.shape[1])})
            result = self.sampler_qnn.run(bind)
            probs = np.array([result.get_probabilities().get(i, 0.0) for i in range(2)])
            probs_list.append(probs)

        probs = np.stack(probs_list, axis=1)  # (batch, num_patches, 2)

        # Use the averaged probabilities as input to the estimator
        avg_probs = probs.mean(axis=1)  # (batch, 2)
        est_bind = {
            self.est_input: np.mean(avg_probs[:, 0]),
            self.est_weight: np.mean(avg_probs[:, 1]),
        }
        expectation = self.estimator_qnn.run(est_bind).expectation_value
        return probs, expectation

__all__ = ["HybridSamplerQNN"]
