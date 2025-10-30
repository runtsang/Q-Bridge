"""Hybrid autoencoder – quantum implementation.

The quantum encoder is built from a QCNN‑inspired variational circuit
(and a quantum kernel) that maps classical data into a latent space.
A simple classical linear decoder refines the latent vector to reconstruct
the input.  The module exposes a single class – ``HybridAutoencoder`` –
which can be instantiated and used in a hybrid training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import Estimator as EstimatorQiskit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
#  Quantum encoder – QCNN style variational circuit
# --------------------------------------------------------------------------- #
def _conv_circuit(num_qubits: int, param_name: str) -> QuantumCircuit:
    """Two‑qubit convolution block used in the QCNN layers."""
    qc = QuantumCircuit(num_qubits, name="conv_block")
    params = qiskit.circuit.ParameterVector(param_name, num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.compose(
            RealAmplitudes(2, reps=1, name=f"{param_name}_{i}"),
            [i, i + 1],
            inplace=True,
        )
    return qc


def _pool_circuit(num_qubits: int, param_name: str) -> QuantumCircuit:
    """Pooling block – simple measurement‑based reduction."""
    qc = QuantumCircuit(num_qubits, name="pool_block")
    # For simplicity we just apply a CNOT chain that correlates qubits
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def _build_qcnn_circuit(num_qubits: int, layers: int) -> QuantumCircuit:
    """Assemble a QCNN‑style circuit with alternating conv and pool layers."""
    qc = QuantumCircuit(num_qubits)
    for layer in range(layers):
        qc.compose(_conv_circuit(num_qubits, f"c{layer}"), inplace=True)
        qc.compose(_pool_circuit(num_qubits, f"p{layer}"), inplace=True)
    return qc


# --------------------------------------------------------------------------- #
#  HybridAutoencoder class
# --------------------------------------------------------------------------- #
class HybridAutoencoder:
    """
    Hybrid autoencoder with a QCNN‑style quantum encoder and a classical linear decoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int
        Size of the latent representation.
    num_qubits : int
        Number of qubits used in the quantum circuit (must be >= input_dim).
    layers : int
        Number of alternating convolution/pooling layers in the QCNN circuit.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_qubits: int = 8,
        layers: int = 3,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = max(num_qubits, input_dim)

        # Feature map that embeds classical data into the quantum state
        self.feature_map = ZFeatureMap(self.num_qubits)

        # Quantum encoder circuit
        self.encoder_circuit = _build_qcnn_circuit(self.num_qubits, layers)

        # Build EstimatorQNN to evaluate the circuit
        self.estimator = EstimatorQiskit()
        self.qnn = EstimatorQNN(
            circuit=self.feature_map.compose(self.encoder_circuit),
            observables=[qiskit.quantum_info.SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.encoder_circuit.parameters,
            estimator=self.estimator,
        )

        # Classical decoder – a simple linear layer in PyTorch
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

    # ----------------------------------------------------------------------- #
    #  Public API
    # ----------------------------------------------------------------------- #
    def encode(self, data: np.ndarray) -> torch.Tensor:
        """
        Encode a batch of data points into the latent space.

        Parameters
        ----------
        data : np.ndarray of shape (batch, input_dim)

        Returns
        -------
        torch.Tensor of shape (batch, latent_dim)
        """
        # Run the QNN and take the absolute value of the first amplitude
        # as the latent feature.  We repeat for each desired latent dimension.
        latents = []
        for i in range(self.latent_dim):
            # Each latent dimension uses a different measurement observable
            observable = qiskit.quantum_info.SparsePauliOp.from_list(
                [(f"Z{i}"+ "I" * (self.num_qubits - i - 1), 1)]
            )
            qnn = EstimatorQNN(
                circuit=self.feature_map.compose(self.encoder_circuit),
                observables=[observable],
                input_params=self.feature_map.parameters,
                weight_params=self.encoder_circuit.parameters,
                estimator=self.estimator,
            )
            probs = qnn.predict(data)
            latents.append(torch.tensor(probs, dtype=torch.float32))
        return torch.stack(latents, dim=1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Map latent vectors back to the input space."""
        return self.decoder(latents)

    def forward(self, data: np.ndarray) -> torch.Tensor:
        """Full autoencoder reconstruction."""
        latents = self.encode(data)
        return self.decode(latents)

    # ----------------------------------------------------------------------- #
    #  Utility – kernel matrix using the quantum kernel
    # ----------------------------------------------------------------------- #
    def quantum_kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between ``a`` and ``b`` using the QCNN encoder."""
        kernel = np.zeros((len(a), len(b)), dtype=float)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                # Simple inner‑product of encoded states as kernel
                x_latent = self.encode(x.reshape(1, -1))
                y_latent = self.encode(y.reshape(1, -1))
                kernel[i, j] = torch.dot(x_latent.squeeze(), y_latent.squeeze()).item()
        return kernel

__all__ = ["HybridAutoencoder"]
