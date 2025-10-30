"""FraudDetectionHybridModel - quantum implementation.

This module uses Strawberry Fields to build a photonic circuit and
inherits the graph utilities and noise‑aware estimator pattern from
the reference seeds.  The public API matches the classical
FraudDetectionHybridModel, enabling seamless switching between back‑ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np
import networkx as nx
import itertools

Observable = Callable[[np.ndarray], complex | float]

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool = False) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

# ---------------------------------------------------------------------------

def random_photonic_network(
    samples: int,
    cutoff: int = 4,
) -> Tuple[List[int], sf.Program, List[Tuple[np.ndarray, np.ndarray]], sf.Program]:
    """Generate a random 2‑mode photonic program and training data."""
    target_prog = sf.Program(2)
    with target_prog.context as q:
        # Random Gaussian unitary
        Sgate(np.random.uniform(-0.5, 0.5)) | q[0]
        Sgate(np.random.uniform(-0.5, 0.5)) | q[1]
        BSgate(np.random.uniform(0, np.pi / 2), np.random.uniform(0, 2 * np.pi)) | (q[0], q[1])
    backend = sf.backends.FockBackend(n_modes=2, cutoff_dim=cutoff)
    target_unitary = backend.get_unitary(target_prog)
    # training data: random input states and their transformed versions
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        # random input statevector (normalized)
        vec = np.random.normal(size=(cutoff ** 2,)) + 1j * np.random.normal(size=(cutoff ** 2,))
        vec = vec / np.linalg.norm(vec)
        out = target_unitary @ vec
        dataset.append((vec, out))
    return [2, 2], target_prog, dataset, target_prog

def feedforward(
    program: sf.Program,
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    cutoff: int = 4,
) -> List[List[np.ndarray]]:
    stored: List[List[np.ndarray]] = []
    backend = sf.backends.FockBackend(n_modes=2, cutoff_dim=cutoff)
    for sample, _ in samples:
        state = backend.run(program, statevector=sample).statevector
        stored.append([state])
    return stored

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# Estimator utilities ---------------------------------------------------------

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized
    Strawberry Fields circuit.  Observables are callables that accept a
    statevector and return a scalar."""
    def __init__(self, program: sf.Program, cutoff: int = 4):
        self.program = program
        self.backend = sf.backends.FockBackend(n_modes=2, cutoff_dim=cutoff)

    def _run(self, params: Sequence[float]) -> np.ndarray:
        # The parameters are applied via the program's parametric gates.
        state = self.backend.run(self.program, args={}).statevector
        return state

    def evaluate(self, observables: Iterable[Observable], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = self._run(values)
            row = [obs(state) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

# Shared class ---------------------------------------------------------------

class FraudDetectionHybridModel:
    """Quantum counterpart of the classical FraudDetectionHybridModel.

    Provides methods that mirror the classical API: ``build`` returns a
    Strawberry Fields ``Program``, ``random_network`` yields a random
    unitary and training data, ``feedforward`` propagates a batch of
    states, ``fidelity_adjacency`` builds a similarity graph, and
    ``evaluate`` runs the circuit and returns noisy expectation values.
    """
    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.arch = list(qnn_arch)

    def build(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> sf.Program:
        return build_fraud_detection_program(input_params, layers)

    def random_network(self, samples: int, cutoff: int = 4) -> Tuple[List[int], sf.Program, List[Tuple[np.ndarray, np.ndarray]], sf.Program]:
        return random_photonic_network(samples, cutoff=cutoff)

    def feedforward(
        self,
        program: sf.Program,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
        cutoff: int = 4,
    ) -> List[List[np.ndarray]]:
        return feedforward(program, samples, cutoff=cutoff)

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        program: sf.Program,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        cutoff: int = 4,
    ) -> List[List[complex]]:
        estimator = FastEstimator(program, cutoff=cutoff) if shots is not None else FastBaseEstimator(program, cutoff=cutoff)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)
