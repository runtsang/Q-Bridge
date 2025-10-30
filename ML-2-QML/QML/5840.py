"""
Hybrid quantum autoencoder‑classifier.

Provides two complementary circuit factories and a wrapper that
allows easy evaluation of the joint model.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit feature encoding and variational parameters.

    Returns:
        circuit: the quantum circuit
        encoding: parameter vector for data encoding
        weights: parameter vector for variational layers
        observables: Z basis observables on each qubit
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for idx, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[idx], qubit)

    w_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[w_idx], qubit)
            w_idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables


def build_autoencoder_circuit(num_latent: int, num_trash: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a quantum autoencoder based on a swap‑test style circuit.

    Returns:
        circuit: the autoencoder circuit
        encoding: no data‑encoding parameters (empty)
        weights: variational parameters of the RealAmplitudes ansatz
        observables: measurement on the auxiliary qubit
    """
    total = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(total, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Feature ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, list(range(num_latent + num_trash)), inplace=True)

    # Swap‑test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    # Observables
    observables = [SparsePauliOp("Z")]
    return qc, [], list(ansatz.parameters), observables


class HybridQuantumModel:
    """
    Wrapper that chains an autoencoder and a classifier circuit.

    The autoencoder acts as a feature extractor, and its output qubits
    feed into the classifier circuit.  The model is fully parameterised
    and compatible with the classical QNN interface.
    """
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        num_classifier_qubits: int,
        classifier_depth: int,
    ) -> None:
        self.autoencoder, _, ae_weights, _ = build_autoencoder_circuit(num_latent, num_trash)
        self.classifier, _, clf_weights, _ = build_classifier_circuit(num_classifier_qubits, classifier_depth)

        # Merge circuits: autoencoder first, then classifier on the latent qubits
        self.circuit = QuantumCircuit(self.autoencoder.num_qubits)
        self.circuit.compose(self.autoencoder, inplace=True)
        # re‑index latent qubits to start at 0 for classifier
        latent_range = range(num_latent)
        self.circuit.compose(self.classifier, latent_range, inplace=True)

        self.weights = ae_weights + clf_weights

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_weights(self) -> List[ParameterVector]:
        return self.weights

    def get_observables(self) -> List[SparsePauliOp]:
        # use the classifier observables as the output of the model
        _, _, _, obs = build_classifier_circuit(self.circuit.num_qubits, 0)
        return obs


__all__ = [
    "build_classifier_circuit",
    "build_autoencoder_circuit",
    "HybridQuantumModel",
]
