"""Quantum autoencoder that emulates QCNN layers and uses a swap‑test fidelity loss."""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

# --------------------------------------------------------------------------- #
# Helper circuits that mirror the QCNN construction
# --------------------------------------------------------------------------- #

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution kernel used in the QCNN."""
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


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Build a convolutional layer with pairwise two‑qubit kernels."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits - 1, 2):
        qc.append(conv_circuit(params[param_index:param_index + 3]), [qubits[i], qubits[i + 1]])
        qc.barrier()
        param_index += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling kernel used in the QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Build a pooling layer that maps source qubits onto sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
# Swap‑test based auto‑encoding circuit
# --------------------------------------------------------------------------- #

def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct a circuit that compresses data into `num_latent` qubits
    and evaluates fidelity with a swap‑test."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode data into the first `num_latent` qubits (to be defined externally)
    # The actual feature‑map will be composed later.

    # Ansatz for the latent subspace
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=2), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test with an auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc


# --------------------------------------------------------------------------- #
# Hybrid quantum autoencoder class
# --------------------------------------------------------------------------- #

class HybridQuantumAutoencoder:
    """Variational quantum autoencoder that incorporates QCNN‑style
    convolution and pooling layers and uses a swap‑test fidelity loss."""

    def __init__(self, input_dim: int, latent_dim: int = 4, num_trash: int = 4) -> None:
        if (latent_dim + num_trash) % 2!= 0:
            raise ValueError("latent_dim + num_trash must be even for QCNN layers.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash

        # Feature map to encode classical data
        self.feature_map = ZFeatureMap(input_dim)
        # Build the ansatz that mirrors QCNN layers
        self.ansatz = self._build_ansatz()
        # Full circuit with swap‑test
        self.circuit = self._build_full_circuit()

        # Sampler for expectation evaluation
        self.sampler = Sampler()
        # SamplerQNN instance
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=lambda x: x[0],  # probability of measuring 0 on the ancilla
            output_shape=1,
            sampler=self.sampler,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct a QCNN‑style ansatz using conv and pool layers."""
        num_qubits = self.latent_dim + self.num_trash
        ansatz = QuantumCircuit(num_qubits)

        # First convolution + pooling
        ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
        ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), range(num_qubits), inplace=True)

        # Second convolution + pooling (reduce size by half)
        half = num_qubits // 2
        ansatz.compose(conv_layer(half, "c2"), range(half), inplace=True)
        ansatz.compose(pool_layer(list(range(half // 2)), list(range(half // 2, half)), "p2"), range(half), inplace=True)

        return ansatz

    def _build_full_circuit(self) -> QuantumCircuit:
        """Wrap the feature map, ansatz, and swap‑test into a single circuit."""
        total_qubits = self.input_dim + self.latent_dim + self.num_trash + 1
        qc = QuantumCircuit(total_qubits)

        # Apply feature map on the first `input_dim` qubits
        qc.compose(self.feature_map, range(self.input_dim), inplace=True)

        # Apply ansatz on the next `latent_dim + num_trash` qubits
        start = self.input_dim
        qc.compose(self.ansatz, range(start, start + self.latent_dim + self.num_trash), inplace=True)

        # Swap‑test ancilla
        ancilla = total_qubits - 1
        qc.h(ancilla)
        for i in range(self.num_trash):
            qc.cswap(ancilla, start + self.latent_dim + i, start + self.latent_dim + self.num_trash + i)
        qc.h(ancilla)
        qc.measure(ancilla, 0)
        return qc

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return fidelity values for a batch of classical data."""
        if data.ndim == 1:
            data = data.unsqueeze(0)
        data_np = data.cpu().numpy()
        fidelities = self.qnn.forward(data_np)  # shape (batch, 1)
        return torch.tensor(fidelities, dtype=torch.float32)

    def train(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 50,
        lr: float = 1e-2,
        optimizer_cls=COBYLA,
    ) -> list[float]:
        """Train the variational parameters using a classical optimizer."""
        opt = optimizer_cls(maxiter=2000, tol=1e-5, disp=False)
        history: list[float] = []

        def objective(params: np.ndarray) -> float:
            # Update parameters
            self.qnn.set_weights(params)
            # Compute mean loss = 1 - fidelity
            fidelities = self.forward(data).detach().numpy().flatten()
            loss = np.mean(1.0 - fidelities)
            return loss

        init_params = self.qnn.get_weights()
        res = opt.minimize(objective, init_params, method="COBYLA")
        history.append(res.fun)
        self.qnn.set_weights(res.x)
        return history


__all__ = ["HybridQuantumAutoencoder"]
