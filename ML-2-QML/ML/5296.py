"""Hybrid classical‑quantum classifier that fuses the best of both worlds."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    use_qubits: bool = False,
) -> Tuple[
    nn.Module,
    Iterable[int] | Iterable[object],
    Iterable[int],
    list[object],
]:
    """
    Build a *hybrid* classifier that alternates between classical layers
    and an optional quantum variational block.

    Parameters
    ----------
    num_features : int
        Number of input features (or qubits).
    depth : int
        Number of hidden layers in the feed‑forward part.
    use_qubits : bool, default False
        If ``True`` a quantum block replaces the linear head; otherwise
        a classical linear layer is used.

    Returns
    -------
    network : nn.Module
        A ``nn.Sequential`` that contains the classical backbone and,
        if requested, a quantum module.
    encoding : Iterable[int] | Iterable[object]
        Indices of the input features that are fed to the quantum block
        (a ``ParameterVector`` when ``use_qubits`` is ``True``).
    weight_sizes : Iterable[int]
        Number of trainable parameters in each sub‑module.
    observables : list[object]
        Quantum measurement operators when ``use_qubits`` is ``True``;
        otherwise an empty list.
    """
    # ---------- Classical backbone ----------
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # ---------- Optional quantum block ----------
    if use_qubits:
        # Import lazily to avoid heavy dependencies when not needed.
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        # Encode the data into a rotation on each qubit.
        encoding = ParameterVector("x", num_features)
        weights = ParameterVector("theta", num_features * depth)

        circuit = QuantumCircuit(num_features)
        for param, qubit in zip(encoding, range(num_features)):
            circuit.rx(param, qubit)

        # Apply a depth‑controlled ansatz.
        idx = 0
        for _ in range(depth):
            for qubit in range(num_features):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_features - 1):
                circuit.cz(qubit, qubit + 1)

        # The measurement observables used to read out the expectation
        # values that are *all* (Z)‑like.
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_features - i - 1))
            for i in range(num_features)
        ]

        # Wrap the quantum circuit in a Torch-compatible layer.
        class QuantumBlock(nn.Module):
            def __init__(self, qc: QuantumCircuit) -> None:
                super().__init__()
                self.qc = qc
                # Backend placeholder; users may set a backend later.
                self.backend = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.backend is None:
                    raise RuntimeError(
                        "Quantum backend must be set before calling forward. "
                        "Use `set_backend` or assign to the `backend` attribute."
                    )
                # In a real implementation this would call a simulator or API.
                # Here we just return a zero tensor for illustrative purposes.
                return torch.zeros(x.shape[0], num_features, device=x.device)

        layers.append(QuantumBlock(circuit))
    else:
        # Classic head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        observables = []

    network = nn.Sequential(*layers)
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
