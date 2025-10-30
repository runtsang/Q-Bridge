"""Hybrid quantum auto‑encoder using a swap‑test circuit.

The quantum auto‑encoder follows the structure in the original Autoencoder
seed but augments the feature embedding with a ZFeatureMap and a RealAmplitudes
ansatz.  It is wrapped in an EstimatorQNN so that it can be trained with
Qiskit Machine Learning optimizers in a manner directly comparable to the
classical counterpart.

Author: gpt-oss-20b
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _swap_test_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a circuit that performs a swap‑test between the latent sub‑space
    and a fresh data register.  The ancilla qubit is measured to
    extract the fidelity between encoded and reconstructed states."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode latent qubits
    qc.compose(RealAmplitudes(num_latent, reps=3), range(0, num_latent), inplace=True)

    # Trash qubits to prepare for swap test
    for i in range(num_trash):
        qc.cx(num_latent + i, num_latent + num_trash + i)

    # Swap‑test ancilla
    ancilla = num_latent + 2 * num_trash
    qc.h(ancilla)
    for i in range(num_trash):
        qc.cswap(ancilla, num_latent + i, num_latent + num_trash + i)
    qc.h(ancilla)
    qc.measure(ancilla, cr[0])

    return qc


def HybridQuantumAutoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    *,
    feature_dim: int = 8,
) -> EstimatorQNN:
    """Return a QNN that implements the hybrid quantum auto‑encoder.

    Parameters
    ----------
    num_latent : int
        Size of the latent register (the “compressed” representation).
    num_trash : int
        Auxiliary qubits used to perform the swap‑test.
    feature_dim : int
        Dimension of the classical input – must match the feature map size.
    """
    algorithm_globals.random_seed = 42

    # Feature embedding
    feature_map = ZFeatureMap(feature_dim)
    feature_map.decompose()  # avoid unnecessary nesting

    # Core auto‑encoder ansatz
    autoenc_circuit = _swap_test_circuit(num_latent, num_trash)

    # Full circuit
    circuit = QuantumCircuit(feature_dim + autoenc_circuit.num_qubits)
    # Embed classical data
    circuit.compose(feature_map, range(feature_dim), inplace=True)
    # Append auto‑encoder part
    circuit.compose(autoenc_circuit, range(feature_dim), inplace=True)

    # Observable that reads out the measured ancilla
    observable = SparsePauliOp.from_list([("Z" + "I" * (feature_dim + autoenc_circuit.num_qubits - 1), 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=autoenc_circuit.parameters,
        estimator=estimator,
    )
    return qnn


# Example of a simple training loop using a COBYLA optimiser
def train_qautoencoder(
    qnn: EstimatorQNN,
    data: np.ndarray,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train the quantum auto‑encoder with a classical optimiser."""
    from qiskit_machine_learning.optimizers import COBYLA
    opt = COBYLA(maxiter=epochs)
    loss_history: list[float] = []

    def cost_fn(params):
        qnn.set_weights(params)
        preds = qnn.predict(data)
        # The output is a probability; use 1 - fidelity as loss
        loss = np.mean((preds - 1.0) ** 2)
        loss_history.append(loss)
        return loss

    opt.optimize(
        len(qnn.weights),
        cost_fn,
        initial_point=np.random.random(len(qnn.weights)),
    )
    return loss_history


__all__ = ["HybridQuantumAutoencoder", "train_qautoencoder"]
