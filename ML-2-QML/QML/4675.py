"""Quantum autoencoder combining QCNN‑style layers and a quantum LSTM‑like block."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def HybridAutoencoder(
    n_qubits: int = 4,
    latent_dim: int | None = None,
    circuit_depth: int = 3,
) -> SamplerQNN:
    """Return a SamplerQNN that acts as a quantum autoencoder.

    The circuit consists of:
        * a ZFeatureMap that encodes classical data,
        * a RealAmplitudes ansatz (QCNN‑style),
        * a simple quantum LSTM‑like block (sequence of Ry and CNOTs),
    and interprets the measurement outcomes as latent embeddings.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    latent_dim = latent_dim or n_qubits  # default to number of qubits

    # Feature map and ansatz
    feature_map = ZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=circuit_depth)

    # Quantum LSTM‑style block
    lstm_params = ParameterVector("lstm", length=2 * n_qubits)

    # Build the full circuit
    qc = QuantumCircuit(n_qubits)
    qc.append(feature_map, range(n_qubits))
    qc.append(ansatz, range(n_qubits))

    # LSTM‑style rotations and CNOT connections
    for i in range(n_qubits):
        qc.ry(lstm_params[i], i)
        if i < n_qubits - 1:
            qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(lstm_params[n_qubits + i], i)

    # Interpretation: expectation value of Pauli‑Z on each qubit
    def interpret(results: np.ndarray) -> np.ndarray:
        # results shape: (shots, n_qubits)
        probs = results.mean(axis=0)  # probability of measuring |0>
        return 1.0 - 2.0 * probs  # <Z> expectation

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters + lstm_params,
        interpret=interpret,
        output_shape=(latent_dim,),
        sampler=sampler,
    )
    return qnn


__all__ = ["HybridAutoencoder"]
