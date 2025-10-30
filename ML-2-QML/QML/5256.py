"""
QuanvolutionHybrid – Quantum implementation

This module implements a quantum‑enhanced quanvolution pipeline that
mirrors the classical counterpart.  Each image patch is encoded into a
parameterised quantum circuit, evaluated with a Qiskit EstimatorQNN,
and the resulting measurement statistics are fed into a quantum
classifier built with the incremental data‑uploading ansatz from
QuantumClassifierModel.  Graph‑based fidelity adjacency is preserved
by computing overlaps of the resulting quantum states.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StateEstimator

# --- Helper utilities ---------------------------------------------------------

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure states."""
    return abs((a.data.conj().T @ b.data)[0, 0]) ** 2


def fidelity_adjacency(
    states: Iterable[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from quantum state fidelities."""
    graph = nx.Graph()
    states = list(states)
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector, List[SparsePauliOp]]:
    """Incremental data‑uploading circuit used for classification."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return qc, encoding, weights, observables


# --- Main hybrid class ---------------------------------------------------------

class QuanvolutionHybrid:
    """
    Quantum quanvolution with graph‑based state adjacency and a quantum classifier head.

    Parameters
    ----------
    patch_size : int
        Size of the square patch to encode (default 2).
    stride : int
        Stride between patches (default 2).
    num_qubits : int
        Number of qubits per patch circuit.
    classifier_depth : int
        Depth of the classifier ansatz.
    graph_threshold : float
        Fidelity threshold for graph edge creation.
    graph_secondary : float | None
        Secondary threshold for weighted edges.
    """

    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        num_qubits: int = 4,
        classifier_depth: int = 2,
        graph_threshold: float = 0.8,
        graph_secondary: float | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.num_qubits = num_qubits
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

        # Quantum device and estimator
        self.estimator = StateEstimator()
        self.classifier_circuit, self.enc_params, self.cls_weights, self.cls_observables = build_classifier_circuit(
            num_qubits, classifier_depth
        )
        self.classifier_qnn = EstimatorQNN(
            circuit=self.classifier_circuit,
            observables=self.cls_observables,
            input_params=self.enc_params,
            weight_params=self.cls_weights,
            estimator=self.estimator,
        )

    def _encode_patch(self, patch: torch.Tensor) -> Tuple[QuantumCircuit, List[float]]:
        """
        Encode a single image patch into a parameterised quantum circuit.

        Parameters
        ----------
        patch : torch.Tensor
            Patch tensor of shape (batch, patch_size, patch_size).

        Returns
        -------
        Tuple[QuantumCircuit, List[float]]
            The quantum circuit and a list of parameter values for encoding.
        """
        # Flatten patch and normalise to [0, π]
        flat = patch.view(-1).cpu().numpy()
        params = np.pi * flat / flat.max() if flat.max() > 0 else np.zeros_like(flat)
        qc = QuantumCircuit(self.num_qubits)
        for i, param in enumerate(params):
            qc.ry(param, i % self.num_qubits)
        return qc, params.tolist()

    def _sample_patch_state(self, qc: QuantumCircuit, params: List[float]) -> Statevector:
        """Return the statevector after applying the parameterised circuit."""
        bound_qc = qc.bind_parameters(dict(zip(self.enc_params, params)))
        return Statevector(bound_qc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum quanvolution and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 2) from the quantum classifier.
        """
        batch_size = x.size(0)
        patch_feats: List[List[Statevector]] = []

        # Process each patch per example
        for b in range(batch_size):
            image = x[b, 0, :, :]  # (28, 28)
            patches = []
            for r in range(0, 28, self.stride):
                for c in range(0, 28, self.stride):
                    patch = image[r : r + self.patch_size, c : c + self.patch_size]
                    qc, params = self._encode_patch(patch.unsqueeze(0))
                    state = self._sample_patch_state(qc, params)
                    patches.append(state)
            patch_feats.append(patches)

        # Graph adjacency per sample
        logits_batch = []
        for feats in patch_feats:
            graph = fidelity_adjacency(
                feats,
                self.graph_threshold,
                secondary=self.graph_secondary,
            )
            # Use graph degree vector as auxiliary features
            degrees = np.array([d for _, d in graph.degree(weight="weight")])
            # Encode degrees into the classifier circuit
            deg_params = (np.pi * degrees / degrees.max()).tolist() if degrees.max() > 0 else np.zeros_like(degrees).tolist()
            bound_qc = self.classifier_circuit.bind_parameters(
                dict(zip(self.enc_params, deg_params))
            )
            # Evaluate expectation values
            exp_vals = self.estimator.run(bound_qc, self.cls_observables).result().values
            logits_batch.append(torch.tensor(exp_vals, dtype=torch.float32))

        logits = torch.stack(logits_batch)  # (batch, 2)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
