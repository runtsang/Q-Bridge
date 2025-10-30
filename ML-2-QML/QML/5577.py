"""Hybrid quantum autoencoder with swap‑test ansatz, SamplerQNN latent extractor,
and EstimatorQNN regression head.

The quantum module mirrors the classical structure but uses a parameterised
circuit that encodes data, applies a RealAmplitudes ansatz, and performs a
swap‑test to compare latent and trash registers.  A `SamplerQNN` extracts
latent state‑vectors and builds a fidelity graph using the same
`GraphQNN.fidelity_adjacency` utility.  An optional `EstimatorQNN`
provides a regression head for quantum‑enhanced predictions.

Public API:
    * `HybridQuantumAutoencoder` – class exposing `forward`, `latent_graph`,
      and `regression` methods.
    * `build_classifier_circuit` – helper for quantum classification.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# Import graph utilities from the shared GraphQNN module
try:
    from.GraphQNN import fidelity_adjacency as _fidelity_adjacency
except Exception:
    from GraphQNN import fidelity_adjacency as _fidelity_adjacency


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Return a simple variational classifier circuit and metadata."""
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

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class HybridQuantumAutoencoder:
    """Quantum autoencoder mirroring the classical counterpart."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 5,
        use_estimator: bool = False,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.use_estimator = use_estimator

        algorithm_globals.random_seed = 42
        self.sampler = StatevectorSampler()
        if use_estimator:
            self.estimator = StatevectorEstimator()
        else:
            self.estimator = None

        self.circuit = self._build_circuit()
        # SamplerQNN extracts the latent register post swap‑test
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        if use_estimator:
            self.estimator_qnn = EstimatorQNN(
                circuit=self.circuit,
                observables=[SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])],
                input_params=[],
                weight_params=self.circuit.parameters,
                estimator=self.estimator,
            )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the swap‑test autoencoder ansatz."""
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode data into latent + trash registers
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        circuit.barrier()

        aux = self.num_latent + 2 * self.num_trash  # auxiliary qubit
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    def forward(self, inputs: List[Statevector]) -> List[Statevector]:
        """
        Forward pass: for each input statevector, set the data register,
        execute the circuit, and return the latent statevector.

        Parameters
        ----------
        inputs : List[Statevector]
            List of input statevectors to encode.

        Returns
        -------
        List[Statevector]
            List of latent statevectors extracted by the SamplerQNN.
        """
        # Build a circuit with data encoding using RawFeatureVector
        data_circuits = []
        for idx, sv in enumerate(inputs):
            circ = QuantumCircuit(self.circuit.num_qubits)
            # Use RawFeatureVector to embed amplitudes
            raw = RawFeatureVector(sv.data)
            circ.compose(raw, range(self.num_latent), inplace=True)
            circ.compose(self.circuit, inplace=True)
            data_circuits.append(circ)

        # Execute all circuits on the simulator
        all_states = []
        for circ in data_circuits:
            result = self.sampler.run(circ).result()
            all_states.append(Statevector(result.get_statevector(circ)))
        return all_states

    def latent_graph(self, inputs: List[Statevector], threshold: float = 0.9) -> nx.Graph:
        """Build a fidelity graph from the latent statevectors."""
        latents = self.forward(inputs)
        # Convert to numpy arrays
        matrices = [np.array(l.data).reshape(-1) for l in latents]
        norms = np.linalg.norm(matrices, axis=1, keepdims=True) + 1e-12
        normalized = matrices / norms
        dot_matrix = normalized @ normalized.T
        graph = nx.Graph()
        graph.add_nodes_from(range(len(latents)))
        for i in range(len(latents)):
            for j in range(i + 1, len(latents)):
                fid = dot_matrix[i, j] ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph

    def regression(self, inputs: List[Statevector]) -> List[float]:
        """
        Apply the EstimatorQNN regression head to the latent states.

        Parameters
        ----------
        inputs : List[Statevector]
            Input statevectors.

        Returns
        -------
        List[float]
            Predicted scalar values.
        """
        if not self.use_estimator:
            raise RuntimeError("EstimatorQNN head not enabled for this instance.")
        # Build circuits with data encoding
        data_circuits = []
        for sv in inputs:
            circ = QuantumCircuit(self.circuit.num_qubits)
            raw = RawFeatureVector(sv.data)
            circ.compose(raw, range(self.num_latent), inplace=True)
            circ.compose(self.circuit, inplace=True)
            data_circuits.append(circ)
        results = self.estimator_qnn.run(data_circuits).result()
        return [float(res) for res in results.get_counts(data_circuits)]

__all__ = [
    "HybridQuantumAutoencoder",
    "build_classifier_circuit",
]
