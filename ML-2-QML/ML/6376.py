"""Hybrid classifier combining classical feed‑forward and quantum circuit simulation.

This module defines the :class:`HybridClassifier` class that integrates a PyTorch
neural network with a Qiskit variational circuit.  The classical sub‑network
mirrors the depth‑controlled architecture from the original
``QuantumClassifierModel`` seed, while the quantum layer is built from the
incremental data‑uploading ansatz.  The two components are jointly trained
via a simple concatenated feature space, providing a ready‑to‑use research
prototype.

The design intentionally mirrors the EstimatorQNN example: a lightweight
regressor is used as the classical surrogate for the quantum layer,
allowing rapid experimentation on CPU while still exposing the full
quantum circuit for simulation or deployment on a backend.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator as StatevectorEstimator


class HybridClassifier(nn.Module):
    """
    A hybrid classical‑quantum classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of layers in both the classical and quantum ansatz.
    num_qubits : int
        Number of qubits used in the quantum circuit.
    device : str, optional
        PyTorch device (``'cpu'`` or ``'cuda'``).

    Notes
    -----
    The classical network is a stack of ``Linear → ReLU`` blocks followed by a
    linear head.  The quantum circuit implements the incremental data‑uploading
    ansatz from the original seed.  During a forward pass the input is fed to
    both sub‑networks; the resulting feature vectors are concatenated and
    projected to the two‑class logits.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        num_qubits: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        # Classical sub‑network
        self.classical_net = self._build_classical_net(num_features, depth)
        self.classical_net.to(device)

        # Quantum circuit and helper objects
        (
            self.quantum_circuit,
            self.encoding,
            self.weight_params,
            self.observables,
        ) = self._build_quantum_circuit(num_qubits, depth)

        # Estimator for expectation values
        self.estimator = StatevectorEstimator()
        self.final_layer = nn.Linear(num_features + len(self.observables), 2).to(device)

    # ------------------------------------------------------------------ #
    # Classical network construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_classical_net(num_features: int, depth: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_features, num_features))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    # Quantum circuit construction (incremental data‑uploading ansatz)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_quantum_circuit(num_qubits: int, depth: int) -> Tuple[
        QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]
    ]:
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
            SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)])
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    # ------------------------------------------------------------------ #
    # Quantum feature extraction
    # ------------------------------------------------------------------ #
    def _quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation values of the observables for each sample.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, num_qubits)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, num_qubits)`` containing the
            expectation values.
        """
        x_np = x.detach().cpu().numpy()
        exp_vals = []

        for obs in self.observables:
            # Bind parameters for each sample and evaluate
            batch_vals = []
            for sample in x_np:
                param_bindings = dict(zip([str(p) for p in self.encoding], sample))
                bound_qc = self.quantum_circuit.bind_parameters(param_bindings)
                sv = Statevector.from_instruction(bound_qc)
                exp = sv.expectation_value(obs).real
                batch_vals.append(exp)
            exp_vals.append(batch_vals)

        return torch.tensor(exp_vals).T.to(self.device)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the hybrid model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, num_features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        # Classical path
        cls_out = self.classical_net(x)

        # Quantum path (requires that num_features == num_qubits)
        qfeat = self._quantum_features(x)

        # Concatenate and project to logits
        combined = torch.cat([cls_out, qfeat], dim=1)
        logits = self.final_layer(combined)
        return logits


__all__ = ["HybridClassifier"]
