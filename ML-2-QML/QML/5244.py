from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info.operators.base_operator import BaseOperator


# -------------------------------------------------------------------------
# Helper circuits for the QCNN‑style ansatz
# -------------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    pair_idx = 0
    for i in range(0, num_qubits - 1, 2):
        qc.compose(conv_circuit(params[pair_idx * 3 : (pair_idx + 1) * 3]), [i, i + 1], inplace=True)
        qc.barrier()
        pair_idx += 1
    return qc


def pool_layer(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    pair_idx = 0
    for i in range(0, num_qubits - 1, 2):
        qc.compose(pool_circuit(params[pair_idx * 3 : (pair_idx + 1) * 3]), [i, i + 1], inplace=True)
        qc.barrier()
        pair_idx += 1
    return qc


# -------------------------------------------------------------------------
# Quantum sampler class
# -------------------------------------------------------------------------
class SamplerQNNGen128:
    """Quantum sampler that produces a 128‑class probability distribution.

    The ansatz is a 7‑qubit QCNN‑style circuit consisting of three
    convolution–pooling pairs.  A ZFeatureMap encodes the two input
    features into the computational basis.  The sampler returns a
    probability vector of length 128 (2⁷).
    """

    def __init__(self) -> None:
        # Parameters for the feature map (inputs) and the ansatz (weights)
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 54)  # 3 conv + 3 pool layers

        # Build the feature‑map + ansatz circuit
        feature_map = ZFeatureMap(7)
        ansatz = self._build_ansatz()

        circuit = QuantumCircuit(7)
        circuit.compose(feature_map, range(7), inplace=True)
        circuit.compose(ansatz, range(7), inplace=True)

        # Sampler primitive
        self.sampler = StatevectorSampler()

        # Wrap with the Qiskit Machine Learning SamplerQNN
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    # ---------------------------------------------------------------------
    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the 7‑qubit QCNN ansatz with three conv–pool layers."""
        # Slice the weight vector into segments for each layer
        segs = [
            self.weight_params[0:9],   # conv1
            self.weight_params[9:18],  # pool1
            self.weight_params[18:27],  # conv2
            self.weight_params[27:36],  # pool2
            self.weight_params[36:45],  # conv3
            self.weight_params[45:54],  # pool3
        ]

        qc = QuantumCircuit(7)
        qc.compose(conv_layer(7, segs[0]), [i for i in range(7)], inplace=True)
        qc.compose(pool_layer(7, segs[1]), [i for i in range(7)], inplace=True)
        qc.compose(conv_layer(7, segs[2]), [i for i in range(7)], inplace=True)
        qc.compose(pool_layer(7, segs[3]), [i for i in range(7)], inplace=True)
        qc.compose(conv_layer(7, segs[4]), [i for i in range(7)], inplace=True)
        qc.compose(pool_layer(7, segs[5]), [i for i in range(7)], inplace=True)
        return qc

    # ---------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass that delegates to the Qiskit SamplerQNN."""
        return self.qnn(inputs)

    # ---------------------------------------------------------------------
    def evaluate(
        self,
        observables: list[BaseOperator],
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[complex]]:
        """
        Evaluate the sampler with optional shot noise, mimicking the
        FastEstimator pattern from reference 4.  When ``shots`` is
        provided, Gaussian noise with variance 1/shots is added to each
        expectation value.
        """
        # Deterministic evaluation
        results = self.qnn.evaluate(observables, parameter_sets)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [rng.normal(val, max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


def SamplerQNN() -> SamplerQNNGen128:
    """Convenience factory that mirrors the original API."""
    return SamplerQNNGen128()


__all__ = ["SamplerQNNGen128", "SamplerQNN"]
