"""Hybrid classical‑quantum classifier combining feed‑forward network with variational quantum circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[SparsePauliOp]]:
    """
    Construct a hybrid classifier: classical feed‑forward network plus metadata for quantum feature extraction.

    Returns:
        classifier: nn.Sequential that outputs logits.
        encoding_params: list of encoding parameter names (for reference).
        weight_sizes: list of number of trainable parameters in each classical layer.
        observables: list of Pauli‑Z observables used for quantum feature extraction.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    classifier = nn.Sequential(*layers)

    # Quantum metadata placeholders (to be filled by the quantum module)
    encoding_params: List[int] = []  # type: ignore
    observables: List[SparsePauliOp] = []  # type: ignore

    return classifier, encoding_params, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Hybrid classifier that augments a classical feed‑forward network with a variational quantum circuit
    to extract expectation‑value features. The quantum circuit is evaluated on a state‑vector simulator.
    """

    def __init__(self, num_features: int, depth: int, num_qubits: int, quantum_depth: int):
        super().__init__()
        self.classifier, _, self.weight_sizes, _ = build_classifier_circuit(num_features, depth)
        self.num_qubits = num_qubits
        self.quantum_depth = quantum_depth

        # Linear map from raw features to quantum encoding
        self.enc_mapper = nn.Linear(num_features, num_qubits)

        # Quantum circuit construction
        self._build_quantum_circuit()

        # Final classifier that fuses classical and quantum features
        self.final = nn.Linear(num_features + num_qubits, 2)

        self.backend = AerSimulator()

    def _build_quantum_circuit(self) -> None:
        """Builds the variational quantum circuit and stores metadata."""
        self.encoding_params = ParameterVector("x", self.num_qubits)
        self.weight_params = ParameterVector("theta", self.num_qubits * self.quantum_depth)

        self.circuit = QuantumCircuit(self.num_qubits)
        for i, param in enumerate(self.encoding_params):
            self.circuit.rx(param, i)

        idx = 0
        for _ in range(self.quantum_depth):
            for i in range(self.num_qubits):
                self.circuit.ry(self.weight_params[idx], i)
                idx += 1
            for i in range(self.num_qubits - 1):
                self.circuit.cz(i, i + 1)

        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

    def _quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum circuit for each sample and return expectation values."""
        with torch.no_grad():
            enc = self.enc_mapper(x)  # batch x num_qubits
            quantum_features = []
            for enc_sample in enc:
                param_bindings = {
                    self.encoding_params[i]: float(enc_sample[i].item())
                    for i in range(self.num_qubits)
                }
                # Bind all weight parameters to zero (static for inference)
                for w in self.weight_params:
                    param_bindings[w] = 0.0
                bound = self.circuit.bind_parameters(param_bindings)
                result = self.backend.run(bound).result()
                state = Statevector(result.get_statevector(bound))
                exp_vals = [state.expectation_value(obs).real for obs in self.observables]
                quantum_features.append(exp_vals)
            return torch.tensor(quantum_features, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: classical features + quantum expectation values → logits."""
        classical = self.classifier(x)
        quantum = self._quantum_features(x)
        combined = torch.cat([classical, quantum], dim=-1)
        return self.final(combined)


__all__ = ["HybridClassifier", "build_classifier_circuit"]
