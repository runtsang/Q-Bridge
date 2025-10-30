"""Quantum hybrid autoencoder leveraging a QCNN ansatz for compression."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper functions to build a QCNN‑style ansatz
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    param_idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _conv_circuit(params[param_idx : param_idx + 3])
        qc.append(sub.to_instruction(), [q1, q2])
        param_idx += 3
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    qubits = list(range(num_qubits))
    param_idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _pool_circuit(params[param_idx : param_idx + 3])
        qc.append(sub.to_instruction(), [q1, q2])
        param_idx += 3
    return qc

def _build_qcnn_ansatz(num_qubits: int) -> QuantumCircuit:
    """Construct a QCNN ansatz with alternating convolution and pooling layers."""
    qc = QuantumCircuit(num_qubits)

    # First convolution and pooling
    qc.append(_conv_layer(num_qubits, "c1").to_instruction(), range(num_qubits))
    qc.append(_pool_layer(num_qubits, "p1").to_instruction(), range(num_qubits))

    # Second convolution and pooling (reduces effective degrees of freedom)
    qc.append(_conv_layer(num_qubits, "c2").to_instruction(), range(num_qubits))
    qc.append(_pool_layer(num_qubits, "p2").to_instruction(), range(num_qubits))

    return qc

# --------------------------------------------------------------------------- #
# Hybrid quantum autoencoder
# --------------------------------------------------------------------------- #
class HybridQuantumAutoencoder:
    """Quantum autoencoder that uses a QCNN ansatz to produce a latent vector."""
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        algorithm_globals.random_seed = 42
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Feature map matching the input dimensionality
        self.feature_map = ZFeatureMap(input_dim)

        # QCNN ansatz
        self.ansatz = _build_qcnn_ansatz(input_dim)

        # Observables: Z on the first `latent_dim` qubits
        self.observables = []
        for i in range(latent_dim):
            op_str = "I" * i + "Z" + "I" * (input_dim - i - 1)
            self.observables.append(SparsePauliOp.from_list([(op_str, 1)]))

        # Backend estimator
        self.estimator = Estimator()

        # Quantum neural network
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

        # Classical decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, inputs: np.ndarray) -> torch.Tensor:
        """Return the quantum latent representation as a torch tensor."""
        result = self.qnn.evaluate_batch(inputs)
        return torch.tensor(result, dtype=torch.float32)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: np.ndarray) -> torch.Tensor:
        return self.decode(self.encode(inputs))

    def train(self, data: np.ndarray, *, epochs: int = 50, lr: float = 1e-3):
        """Jointly train the QCNN ansatz and the classical decoder."""
        opt = torch.optim.Adam(list(self.qnn.parameters()) + list(self.decoder.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            latents = self.encode(data)
            recon = self.decode(latents)
            loss = loss_fn(recon, torch.tensor(data, dtype=torch.float32))
            opt.zero_grad()
            loss.backward()
            opt.step()

__all__ = ["HybridQuantumAutoencoder"]
