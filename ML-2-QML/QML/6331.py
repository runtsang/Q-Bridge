"""Quantum hybrid autoencoder using swap‑test latent read‑out and QCNN‑style feature map."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def QuantumHybridAutoencoder(
    input_dim: int,
    latent_dim: int = 3,
    trash_qubits: int = 2,
    feature_map_qubits: int | None = None,
    reps: int = 5,
) -> SamplerQNN:
    """
    Returns a :class:`SamplerQNN` that implements an autoencoder circuit.
    The circuit is built from:
    * a QCNN‑style feature map (ZFeatureMap) applied to the input data,
    * a RealAmplitudes ansatz that serves as the encoder,
    * a swap‑test that reads out the latent state,
    * a domain‑wall construction that injects a simple pattern into the trash qubits.

    Parameters
    ----------
    input_dim : int
        Number of classical features.
    latent_dim : int
        Number of qubits representing the latent space.
    trash_qubits : int
        Number of ancillary qubits used during the swap‑test.
    feature_map_qubits : int, optional
        Number of qubits used by the feature map. Defaults to ``latent_dim``.
    reps : int
        Repetitions for the RealAmplitudes ansatz.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # ---- Feature map ----
    feature_map_qubits = feature_map_qubits or latent_dim
    feature_map = ZFeatureMap(feature_map_qubits)

    # ---- Encoder ansatz ----
    def encoder_ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=reps)

    # ---- Swap‑test based latent read‑out ----
    def swap_test(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode data into the first num_latent qubits
        qc.append(encoder_ansatz(num_latent), range(num_latent))

        # Domain‑wall pattern on trash qubits
        for i in range(num_trash):
            qc.x(num_latent + i)
        for i in range(num_trash):
            qc.x(num_latent + num_trash + i)

        # Swap‑test
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # ---- Build full circuit ----
    qc = QuantumCircuit(feature_map_qubits)
    qc.append(feature_map, range(feature_map_qubits))
    qc.compose(swap_test(latent_dim, trash_qubits), range(latent_dim + 2 * trash_qubits + 1), inplace=True)

    # ---- Interpret output as a probability amplitude ----
    def interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=encoder_ansatz(latent_dim + 2 * trash_qubits + 1).parameters,
        interpret=interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


__all__ = ["QuantumHybridAutoencoder"]
