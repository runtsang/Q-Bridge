"""Quantum autoencoder and classifier using Qiskit."""

from __future__ import annotations

from typing import Tuple, Iterable, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as SamplerPrimitive
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def _build_autoencoder_circuit(latent_dim: int, trash_dim: int) -> QuantumCircuit:
    qr = QuantumRegister(latent_dim + 2 * trash_dim + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(latent_dim + trash_dim, reps=5)
    qc.compose(ansatz, range(0, latent_dim + trash_dim), inplace=True)
    qc.barrier()
    aux = latent_dim + 2 * trash_dim
    qc.h(aux)
    for i in range(trash_dim):
        qc.cswap(aux, latent_dim + i, latent_dim + trash_dim + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def _build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

def build_autoencoder_circuit(latent_dim: int = 3, trash_dim: int = 2) -> SamplerQNN:
    """Constructs a variational autoencoder as a SamplerQNN."""
    algorithm_globals.random_seed = 42
    sampler = SamplerPrimitive()
    qc = _build_autoencoder_circuit(latent_dim, trash_dim)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    return _build_classifier_circuit(num_qubits, depth)

class AutoencoderGen089:
    """Quantum variational autoencoder optionally augmented with a classifier ansatz."""
    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        classifier_depth: int = 2,
        use_classifier: bool = False,
    ) -> None:
        algorithm_globals.random_seed = 42
        self.sampler = SamplerPrimitive()
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.classifier_depth = classifier_depth
        self.use_classifier = use_classifier

        self.auto_circuit = _build_autoencoder_circuit(latent_dim, trash_dim)
        if use_classifier:
            self.cls_circuit, self.cls_encoding, self.cls_weights, self.cls_observables = _build_classifier_circuit(
                latent_dim, classifier_depth
            )
            self.auto_circuit.compose(self.cls_circuit, range(0, latent_dim), inplace=True)

        self.qnn = SamplerQNN(
            circuit=self.auto_circuit,
            input_params=[],
            weight_params=self.auto_circuit.parameters,
            interpret=lambda x: x,
            output_shape=2 if use_classifier else 1,
            sampler=self.sampler,
        )
        self.optimizer = COBYLA(maxiter=200)

    def train(self, training_data: np.ndarray, epochs: int = 100, learning_rate: float = 1e-3) -> None:
        """Simple COBYLA‑based training loop over the circuit parameters."""
        best_loss = float("inf")
        best_params = {}
        for _ in range(epochs):
            params = {p: np.random.uniform(-np.pi, np.pi) for p in self.auto_circuit.parameters}
            result = self.sampler.run(self.auto_circuit.bind_parameters(params), shots=1024)
            probs = result.get_counts()
            loss = self._compute_loss(probs)
            if loss < best_loss:
                best_loss = loss
                best_params = params
        self.auto_circuit.assign_parameters(best_params, inplace=True)

    def _compute_loss(self, probs: dict) -> float:
        """Dummy loss: 1 minus probability of measurement outcome '1'."""
        return 1.0 - probs.get("1", 0.0)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return measurement counts for the given inputs."""
        result = self.sampler.run(self.auto_circuit, shots=1024)
        return result.get_counts()

__all__ = [
    "AutoencoderGen089",
    "build_autoencoder_circuit",
    "build_classifier_circuit",
]
