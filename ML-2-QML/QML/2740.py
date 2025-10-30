"""Hybrid quantum autoencoder with classification head."""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Sampler as StatevectorSampler

def build_quantum_autoencoder_circuit(
    num_qubits: int,
    latent_dim: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a quantum autoencoder circuit with swap test and classification observables."""
    # Encoding parameters
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    # Total qubits: data qubits + ancilla for swap test
    total_qubits = num_qubits + 1
    qc = QuantumCircuit(total_qubits)

    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Swap test with ancilla qubit (last qubit)
    ancilla = total_qubits - 1
    qc.h(ancilla)
    for i in range(num_qubits):
        qc.cswap(ancilla, i, i)
    qc.h(ancilla)

    # Observables: ancilla Z for fidelity, data qubit Z for classification
    observables = [SparsePauliOp("Z" + "I" * num_qubits)]  # ancilla
    for i in range(num_qubits):
        pauli = "I" * i + "Z" + "I" * (num_qubits - i - 1)
        observables.append(SparsePauliOp(pauli))
    return qc, list(encoding), list(weights), observables

class HybridAutoencoder:
    """Quantum neural network implementing a hybrid autoencoder with classification."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        depth: int = 3,
        sampler: StatevectorSampler | None = None,
    ) -> None:
        algorithm_globals.random_seed = 42
        self.circuit, self.encoding_params, self.weight_params, self.observables = build_quantum_autoencoder_circuit(
            num_qubits, latent_dim, depth
        )
        self.sampler = sampler or StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.encoding_params,
            weight_params=self.weight_params,
            interpret=lambda x: x,  # raw probabilities
            output_shape=(2,),
            sampler=self.sampler,
        )

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fidelity and classification logits for a batch of inputs."""
        batch_size = inputs.shape[0]
        fidelities = np.zeros(batch_size)
        logits = np.zeros((batch_size, len(self.observables) - 1))
        for i in range(batch_size):
            param_dict = {p: inputs[i, j] for j, p in enumerate(self.encoding_params)}
            bound_circ = self.circuit.bind_parameters(param_dict)
            result = self.sampler.run(bound_circ).result()
            state = Statevector(result.get_statevector())
            # Fidelity from ancilla Z expectation
            ancilla_expect = state.expectation_value(self.observables[0]).real
            fidelities[i] = (1 + ancilla_expect) / 2
            # Classification logits from data qubit Z expectations
            for k, obs in enumerate(self.observables[1:]):
                logits[i, k] = state.expectation_value(obs).real
        return fidelities, logits

__all__ = [
    "HybridAutoencoder",
    "build_quantum_autoencoder_circuit",
]
