"""Unified quantum QCNN mirroring the classical architecture.

The quantum model first embeds the classical input with a Z‑feature map,
compresses it with a quantum autoencoder (swap‑test style), applies a
hierarchical convolution‑pooling circuit, and finally evaluates a Z observable
on the remaining qubit.  The circuit structure parallels the classical
autoencoder, graph pooling, and transformer stages.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

__all__ = ["UnifiedQCNN"]


class UnifiedQCNN:
    """Quantum analogue of the hybrid QCNN architecture."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        conv_qubits: int = 8,
        pool_qubits: int = 4,
        num_classes: int = 1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.estimator = StatevectorEstimator()
        self.feature_map = ZFeatureMap(num_qubits=input_dim)
        self.autoencoder = self._build_autoencoder()
        self.conv_layer = self._build_conv_layer(conv_qubits)
        self.pool_layer = self._build_pool_layer(pool_qubits)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (conv_qubits - 1), 1)])

    # -------------------------------------------------------------------------
    # Circuit builders
    # -------------------------------------------------------------------------
    def _build_autoencoder(self) -> QuantumCircuit:
        """Quantum autoencoder using a swap‑test style circuit."""
        qr = QuantumRegister(self.latent_dim + 2 * self.latent_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Ansatz: RealAmplitudes on latent + trash qubits
        qc.append(RealAmplitudes(self.latent_dim + self.latent_dim, reps=3), range(0, self.latent_dim + self.latent_dim))
        qc.barrier()
        aux = self.latent_dim + 2 * self.latent_dim
        qc.h(aux)
        for i in range(self.latent_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.latent_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _build_conv_layer(self, num_qubits: int) -> QuantumCircuit:
        """Convolutional layer composed of 2‑qubit unitary blocks."""
        def conv_circuit(params):
            c = QuantumCircuit(2)
            c.rz(-np.pi / 2, 1)
            c.cx(1, 0)
            c.rz(params[0], 0)
            c.ry(params[1], 1)
            c.cx(0, 1)
            c.ry(params[2], 1)
            c.cx(1, 0)
            c.rz(np.pi / 2, 0)
            return c

        params = ParameterVector("θ", length=3 * (num_qubits // 2))
        qc = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[3 * (i // 2) : 3 * (i // 2 + 1)])
            qc.append(sub, [i, i + 1])
        return qc

    def _build_pool_layer(self, num_qubits: int) -> QuantumCircuit:
        """Pooling layer that discards half the qubits."""
        def pool_circuit(params):
            c = QuantumCircuit(2)
            c.rz(-np.pi / 2, 1)
            c.cx(1, 0)
            c.rz(params[0], 0)
            c.ry(params[1], 1)
            c.cx(0, 1)
            c.ry(params[2], 1)
            return c

        params = ParameterVector("ϕ", length=3 * (num_qubits // 2))
        qc = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits, 2):
            sub = pool_circuit(params[3 * (i // 2) : 3 * (i // 2 + 1)])
            qc.append(sub, [i, i + 1])
        return qc

    # -------------------------------------------------------------------------
    # Forward evaluation
    # -------------------------------------------------------------------------
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the QCNN on a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, input_dim) with float features.

        Returns
        -------
        np.ndarray
            Array of shape (batch, ) with expectation values of the observable.
        """
        batch_size = inputs.shape[0]
        results = []
        for sample in inputs:
            # Feature map
            circuit = QuantumCircuit(self.input_dim)
            circuit.compose(self.feature_map, range(self.input_dim), inplace=True)
            # Autoencoder
            circuit.compose(self.autoencoder, range(self.latent_dim + 2 * self.latent_dim + 1), inplace=True)
            # Convolution + pooling
            circuit.compose(self.conv_layer, range(self.input_dim), inplace=True)
            circuit.compose(self.pool_layer, range(self.input_dim), inplace=True)
            # Truncate to remaining qubits
            circuit = circuit.decompose()
            # Build EstimatorQNN
            qnn = EstimatorQNN(
                circuit=circuit,
                observables=self.observable,
                input_params=[],
                weight_params=[],
                estimator=self.estimator,
            )
            result = qnn.evaluate(sample.reshape(1, -1))
            results.append(result[0])
        return np.array(results)
