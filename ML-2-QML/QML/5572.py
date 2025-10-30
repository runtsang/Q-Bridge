"""Hybrid fraud‑detection module: quantum side of the hybrid network.

The quantum module is a thin wrapper that exposes the same
parameter container and a single variational circuit that can be used
in a larger hybrid pipeline.  It implements the same logic as the
classical side but in a photonic‑style continuous‑variable circuit
using the Strawberry‑Fields library.  The circuit is built once
and then re‑used for inference and gradient‑based training.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import strawberryfields as sf
from strawberryfields.ops import BSgate, Sgate, Dgate, Rgate, Kgate
import numpy as np
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Parameter container (identical to classical version)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FraudLayerParameters:
    """Parameter container used by the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Helper functions – build quantum program
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clip all continuous‑variable parameters to the same bound as in the
    original photonic seed."""
    return max(-bound, min(bound, value))

def _apply_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a single photonic layer to the provided Strawberry‑Fields modes."""
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry‑Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# --------------------------------------------------------------------------- #
# 3. Quantum net wrapper
# --------------------------------------------------------------------------- #
class FraudDetectionQuantumNet:
    """
    Wrapper around a Strawberry‑Fields program that exposes a simple
    interface for inference and gradient‑based training.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (classical) layer.
    hidden_layers : Iterable[FraudLayerParameters]
        Iterable of subsequent layer parameters.
    cutoff_dim : int, optional
        Truncation dimension for the continuous‑variable simulation.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_layers: Iterable[FraudLayerParameters],
        cutoff_dim: int = 10,
    ) -> None:
        self.program = build_fraud_detection_program(input_params, hidden_layers)
        self.engine = sf.Engine("gaussian", backend_options={"cutoff_dim": cutoff_dim})
        # Pre‑compile the program for faster repeated evaluation
        self.engine.compile(self.program)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the program for a batch of two‑dimensional inputs.

        Parameters
        ----------
        inputs : np.ndarray of shape (batch, 2)
            Classical feature matrix.

        Returns
        -------
        expectations : np.ndarray of shape (batch, 1)
            Expectation value of the photon number operator on mode 0
            for each input.
        """
        # Prepare the input state: displaced vacuum with the given amplitudes
        displacements = inputs.T  # shape (2, batch)
        # Run the program for each input
        expectations = []
        for disp in displacements.T:
            self.program["0"].displace(disp[0], 0)
            self.program["1"].displace(disp[1], 0)
            state = self.engine.run(self.program).state
            # Photon number expectation of mode 0
            exp = state.expectation_value(sf.ops.N(0))
            expectations.append(exp)
        return np.array(expectations).reshape(-1, 1)

# --------------------------------------------------------------------------- #
# 4. Synthetic data utilities (borrowed from GraphQNN)
# --------------------------------------------------------------------------- #
def random_fraud_dataset(
    n_samples: int,
    arch: List[int],
    weight: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate random feature/target pairs using the same linear mapping
    used in :func:`GraphQNN.random_network`."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_samples):
        features = np.random.randn(weight.shape[1])
        target = weight @ features
        dataset.append((features, target))
    return dataset

# --------------------------------------------------------------------------- #
# 5. Graph‑based adjacency for clustering fraud patterns
# --------------------------------------------------------------------------- #
def fraud_graph_from_fidelities(
    states: List[sf.State],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a graph where nodes are quantum states and edges encode
    fidelity‑like similarity.  The fidelity is computed as the absolute
    squared overlap between two pure states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i + 1 :], start=i + 1):
            fid = abs((a.dag() * b)[0, 0]) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionQuantumNet",
    "random_fraud_dataset",
    "fraud_graph_from_fidelities",
    "build_fraud_detection_program",
]
