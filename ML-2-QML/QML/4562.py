"""FastHybridEstimator: quantum implementation for circuit evaluation, auto‑encoding, and fidelity‑based graph construction."""

from __future__ import annotations

from typing import Iterable, Sequence, List, Callable, Optional
import itertools
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler

# --------------------------------------------------------------------------- #
# Quantum auto‑encoder helper
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> Sequence[float]:
    """Convenience wrapper that accepts a list of floats or a 2‑D list."""
    return values if isinstance(values, list) else [values]


def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build the swap‑test auto‑encoder circuit used in the reference."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Feature embedding
    qc.compose(RealAmplitudes(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Domain‑wall preparation
    for i in range(num_trash):
        qc.x(num_latent + num_trash + i)

    # Swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc


def QuantumAutoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    seed: int | None = None,
) -> SamplerQNN:
    """Return a variational quantum neural network that implements the auto‑encoder."""
    algorithm_globals.random_seed = seed or 42
    sampler = StatevectorSampler()
    qc = _autoencoder_circuit(num_latent, num_trash)

    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=(2,),
        sampler=sampler,
    )


# --------------------------------------------------------------------------- #
# Fidelity‑based adjacency (quantum version)
# --------------------------------------------------------------------------- #
def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Unified quantum estimator
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """
    Quantum estimator that can:
    * evaluate expectation values of a parametrised circuit,
    * run a quantum auto‑encoder via :class:`SamplerQNN`,
    * build fidelity‑based adjacency graphs from statevectors.
    """
    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        qnn: SamplerQNN | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.qnn = qnn
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        if self.circuit is None and self.qnn is None:
            raise RuntimeError("No circuit or QNN provided.")
        observables = list(observables or [])
        results: List[List[complex]] = []

        for params in parameter_sets:
            if self.qnn is not None:
                # QNN directly returns output vectors
                out = self.qnn(np.array(params).reshape(1, -1))
                results.append(out.tolist()[0])
            else:
                bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        return results

    def run_autoencoder(
        self,
        input_values: Sequence[float],
    ) -> np.ndarray:
        """Apply the quantum auto‑encoder and return the measurement distribution."""
        if self.qnn is None:
            raise RuntimeError("QNN not configured.")
        return self.qnn(np.array(input_values).reshape(1, -1))[0]

    def graph_fidelity(
        self,
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
    ) -> nx.Graph:
        """Return a fidelity‑based adjacency graph constructed from statevectors."""
        return fidelity_adjacency(states, threshold, secondary=secondary)


__all__ = [
    "QuantumAutoencoder",
    "state_fidelity",
    "fidelity_adjacency",
    "FastHybridEstimator",
]
