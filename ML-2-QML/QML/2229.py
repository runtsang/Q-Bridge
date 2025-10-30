"""Hybrid quantum classifier module that mirrors the classical API while leveraging
a parameterised Qiskit circuit and a quantum kernel based on state‑vector fidelity.

The design intentionally keeps the public interface identical to the classical
``HybridClassifier`` so that side‑by‑side experiments can be performed without
changing downstream code.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterable, Tuple, Sequence

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, SparsePauliOp

# --------------------------------------------------------------------------- #
#  Quantum kernel utilities
# --------------------------------------------------------------------------- #
class QuantumKernel(torch.nn.Module):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap between the two encoded states."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        circuit_x = QuantumCircuit(self.n_wires)
        circuit_y = QuantumCircuit(self.n_wires)

        for i in range(self.n_wires):
            circuit_x.rx(x[0, i].item(), i)
            circuit_y.rx(y[0, i].item(), i)

        state_x = Statevector.from_instruction(circuit_x)
        state_y = Statevector.from_instruction(circuit_y)
        return torch.abs(state_x.inner(state_y).real)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Quantum classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered variational circuit with explicit data encoding.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised ansatz ready for measurement.
    encoding : Iterable[ParameterVector]
        List of data‑encoding parameters.
    weights : Iterable[ParameterVector]
        List of variational parameters.
    observables : list[SparsePauliOp]
        Pauli operators used for read‑out.
    """
    encoding = [f"x_{i}" for i in range(num_qubits)]
    weights = [f"theta_{i}" for i in range(num_qubits * depth)]

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
    return circuit, encoding, weights, observables

# --------------------------------------------------------------------------- #
#  Hybrid quantum classifier class
# --------------------------------------------------------------------------- #
class HybridClassifier(torch.nn.Module):
    """
    Quantum hybrid classifier that exposes the same API as its classical sibling.
    The model consists of a parameterised circuit and, when ``use_kernel`` is
    enabled, a quantum kernel that augments the measurement with a similarity
    score to a small support set.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used by the variational circuit.
    depth : int, default 2
        Depth of the variational layers.
    use_kernel : bool, default False
        If ``True`` a quantum kernel is evaluated against ``support_vectors`` and
        the resulting scalar is appended to the measurement vector before the
        final read‑out.
    support_vectors : torch.Tensor, optional
        Reference points used when ``use_kernel`` is ``True``.  A random set
        is generated if omitted.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        use_kernel: bool = False,
        support_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = Aer.get_backend("statevector_simulator")

        if use_kernel:
            self.kernel = QuantumKernel()
            if support_vectors is None:
                support_vectors = torch.randn(5, num_qubits)
            self.support_vectors = support_vectors
        else:
            self.kernel = None
            self.support_vectors = None

    def _kernel_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kernel similarity of *x* to the support set."""
        feats = torch.stack([self.kernel(x, sv) for sv in self.support_vectors], dim=-1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the circuit and return the expectation values."""
        # bind data to encoding parameters
        param_dict = {param: val.item() for param, val in zip(self.encoding, x)}
        bound = self.circuit.bind_parameters(param_dict)

        job = execute(bound, self.backend)
        result = job.result()
        state = result.get_statevector(bound)

        # compute expectation of each observable
        exp_vals = torch.tensor(
            [abs(state.expectation_value(obs)) for obs in self.observables]
        )

        if self.use_kernel:
            kernel_feat = self._kernel_feature(x)
            exp_vals = torch.cat([exp_vals, kernel_feat], dim=-1)

        return exp_vals

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits derived from the circuit measurement."""
        return self.forward(x)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Convenience wrapper around the module‑level kernel_matrix helper."""
        return kernel_matrix(a, b)
