"""Quantum hybrid autoencoder that fuses a variational circuit, a sampler QNN,
and a graph‑based fidelity regulariser."""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import SPSA

# --------------------------------------------------------------------------- #
# 1. Variational autoencoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Builds a swap‑test based autoencoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the latent subspace
    qc.append(RealAmplitudes(num_latent + num_trash, reps=5), range(num_latent + num_trash))
    qc.barrier()

    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --------------------------------------------------------------------------- #
# 2. Sampler QNN wrapper
# --------------------------------------------------------------------------- #
def quantum_sampler_qnn(num_latent: int, num_trash: int) -> SamplerQNN:
    """Instantiate a Qiskit SamplerQNN that interprets the swap‑test outcome."""
    qc = quantum_autoencoder_circuit(num_latent, num_trash)
    sampler = StatevectorSampler()
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        sampler=sampler,
        interpret=lambda x: x,
        output_shape=2,
    )

# --------------------------------------------------------------------------- #
# 3. Graph utilities (quantum fidelity)
# --------------------------------------------------------------------------- #
def quantum_fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = states[i].fidelity(states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 4. Quantum hybrid autoencoder class
# --------------------------------------------------------------------------- #
class HybridAutoencoder:
    """Quantum autoencoder that trains a variational circuit with graph‑regularised latent states."""
    def __init__(self, num_latent: int, num_trash: int, shots: int = 1024) -> None:
        self.circuit = quantum_autoencoder_circuit(num_latent, num_trash)
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            sampler=self.sampler,
            interpret=lambda x: x,
            output_shape=2,
        )
        self.shots = shots

    def _sample_statevectors(self, n: int) -> list[Statevector]:
        """Return a list of statevectors sampled from the current circuit."""
        vec = Statevector(self.circuit)
        return [vec] * n

    def train(
        self,
        data: Iterable[np.ndarray],
        *,
        epochs: int = 50,
        learning_rate: float = 0.01,
        graph_regularization: bool = True,
        fidelity_threshold: float = 0.9,
        secondary_threshold: float | None = None,
    ) -> list[float]:
        """Train the variational parameters using SPSA."""
        opt = SPSA(max_trials=100, learning_rate=learning_rate)
        history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample in data:
                # Forward pass: obtain probabilities from the sampler
                probs = self.qnn(sample, shots=self.shots)
                recon = np.array(probs, dtype=np.float32)
                target = sample.astype(np.float32)
                loss = np.mean((recon - target) ** 2)

                if graph_regularization:
                    svs = self._sample_statevectors(self.shots)
                    graph = quantum_fidelity_adjacency(
                        svs, fidelity_threshold,
                        secondary=secondary_threshold,
                    )
                    penalty = 0.0
                    for u, v, data in graph.edges(data=True):
                        penalty += np.linalg.norm(svs[u].data - svs[v].data) ** 2
                    loss += 0.01 * penalty

                opt.step(lambda _: loss)  # SPSA step with constant loss placeholder
                epoch_loss += loss
            epoch_loss /= len(data)
            history.append(epoch_loss)
        return history

__all__ = [
    "HybridAutoencoder",
    "quantum_autoencoder_circuit",
    "quantum_sampler_qnn",
]
