"""Hybrid quantum kernel model with a sampler QNN.

The quantum implementation mirrors the classical API.
It uses a variational circuit to encode data, a sampler
circuit to produce probability amplitudes, and a
simple linear classifier on the kernel matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

__all__ = ["HybridKernelModel"]


class KernalAnsatz(QuantumCircuit):
    """A fixed variational circuit that maps 2‑D classical data to a
    quantum state.  The circuit is intentionally shallow so that the
    kernel can be evaluated on a simulator in a few milliseconds.
    """

    def __init__(self, n_wires: int = 2) -> None:
        super().__init__(n_wires)
        self.input_params = ParameterVector("x", n_wires)
        for i in range(n_wires):
            self.ry(self.input_params[i], i)
        # Add a few entangling layers
        self.compose(TwoLocal(n_wires, ["ry"], "cz", reps=1, insert_barriers=False), inplace=True)

    def bind(self, data: np.ndarray) -> "KernalAnsatz":
        """Return a circuit with the input parameters bound to ``data``."""
        return self.bind_parameters(dict(zip(self.input_params, data)))


class Kernel:
    """Quantum kernel that evaluates the overlap between the states
    produced by two instances of :class:`KernalAnsatz`.
    """

    def __init__(self, n_wires: int = 2) -> None:
        self.n_wires = n_wires
        self.base_circuit = KernalAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute value of the overlap of the two encoded states."""
        sampler = StatevectorSampler()
        qc_x = self.base_circuit.bind(x.numpy())
        qc_y = self.base_circuit.bind(y.numpy())
        sv_x = sampler.run(qc_x).result().get_statevector()
        sv_y = sampler.run(qc_y).result().get_statevector()
        fidelity = np.abs(np.vdot(sv_x, sv_y)) ** 2
        return torch.tensor(fidelity, dtype=torch.float32)


class HybridKernelModel:
    """Hybrid quantum kernel model that mimics the classical API.

    Parameters
    ----------
    use_sampler : bool
        If ``True`` a quantum sampler QNN is used to generate a
        probability distribution over the output space before
        computing the kernel.
    """

    def __init__(self, use_sampler: bool = False) -> None:
        self.kernel = Kernel()
        self.use_sampler = use_sampler
        self.sampler: Optional[SamplerQNN] = None
        if use_sampler:
            # Define a simple 2‑qubit sampler circuit
            sampler_circuit = QuantumCircuit(2)
            sampler_circuit.ry(ParameterVector("x", 2)[0], 0)
            sampler_circuit.ry(ParameterVector("x", 2)[1], 1)
            sampler_circuit.cx(0, 1)
            self.sampler = SamplerQNN(
                circuit=sampler_circuit,
                input_params=ParameterVector("x", 2),
                weight_params=ParameterVector("w", 2),
            )
        # Linear classifier with a single weight
        self.classifier_weight = torch.tensor(1.0, requires_grad=True)

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the quantum sampler (if enabled) to each sample."""
        if self.use_sampler and self.sampler is not None:
            # The sampler returns a probability distribution; we flatten it
            probs = self.sampler(data)
            return probs
        return data

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between ``a`` and ``b``."""
        a_t = [self._transform(x) for x in a]
        b_t = [self._transform(y) for y in b]
        return np.array(
            [[self.kernel.forward(a_t[i], b_t[j]).item() for j in range(len(b_t))] for i in range(len(a_t))]
        )

    def predict(self, kernel_mat: np.ndarray) -> np.ndarray:
        """Apply the linear classifier to the kernel matrix."""
        k = torch.from_numpy(kernel_mat).float()
        preds = (self.classifier_weight * k).sum(-1)
        return preds.detach().numpy()
