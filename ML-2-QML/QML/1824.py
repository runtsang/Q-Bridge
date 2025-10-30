"""Quantum kernel module using Pennylane variational circuits."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.operation import Operation


class QuantumKernelMethod:
    """
    Quantum kernel evaluator built on Pennylane's variational circuits.

    Parameters
    ----------
    wires : int
        Number of qubits.
    layers : int, optional
        Depth of the variational circuit.
    base_gate : str, optional
        Primary single‑qubit gate; ``'RY'`` by default.
    backend : str, optional
        Pennylane device name; defaults to ``'default.qubit'``.
    seed : int, optional
        Random seed for circuit initialization.

    Notes
    -----
    The kernel is defined as the absolute square of the overlap between two
    states prepared by encoding inputs ``x`` and ``-y`` into the same circuit,
    mirroring the structure of the original TorchQuantum implementation.
    """

    def __init__(
        self,
        wires: int,
        *,
        layers: int = 2,
        base_gate: str = "RY",
        backend: str = "default.qubit",
        seed: int | None = None,
    ) -> None:
        self.wires = wires
        self.layers = layers
        self.base_gate = base_gate
        self.backend = backend
        self.seed = seed
        self.dev = qml.device(backend, wires=wires, shots=None)
        self._build_kernel()

    def _variational_circuit(self, params: Iterable[float], data: Iterable[float]) -> None:
        """
        Parameterised circuit that encodes classical data and variational parameters.

        Parameters
        ----------
        params
            Flattened list of variational parameters.
        data
            Input data to be encoded.
        """
        for i in range(self.layers):
            for w in range(self.wires):
                # Encoding layer
                qml.apply(
                    getattr(qml, self.base_gate),
                    wires=w,
                    parameters=data[w],
                )
                # Entangling layer
                qml.CNOT(wires=[w, (w + 1) % self.wires])
        # Variational rotation
        if self.base_gate in ("RX", "RY", "RZ"):
            for idx, param in enumerate(params):
                qml.apply(
                    getattr(qml, self.base_gate),
                    wires=idx % self.wires,
                    parameters=param,
                )

    @qnode
    def _kernel_qnode(self, x: Sequence[float], y: Sequence[float]) -> float:
        """
        QNode that returns the absolute overlap between |ψ(x)⟩ and |ψ(-y)⟩.
        """
        # Encode x
        self._variational_circuit(self.params, x)
        # Encode -y
        self._variational_circuit(self.params, [-v for v in y])
        return qml.expval(qml.PauliZ(0))

    @property
    def params(self) -> np.ndarray:
        """
        Lazy initialisation of variational parameters.
        """
        if not hasattr(self, "_params"):
            rng = pnp.random.default_rng(self.seed)
            self._params = rng.standard_normal(self.layers * self.wires)
        return self._params

    def _build_kernel(self) -> None:
        """
        Compile the QNode once to avoid recompilation overhead.
        """
        self.kernel_qnode = self._kernel_qnode

    def __call__(self, x: Sequence[float], y: Sequence[float]) -> float:
        """
        Evaluate the quantum kernel for two input vectors.

        Parameters
        ----------
        x, y : Sequence[float]
            Classical data vectors of the same length.

        Returns
        -------
        float
            Kernel value in [0, 1].
        """
        return abs(self.kernel_qnode(x, y))

    def kernel_matrix(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between datasets ``a`` and ``b``.

        Supports optional batching to keep memory usage in check.

        Parameters
        ----------
        a, b : Sequence[Sequence[float]]
            Sequences of 1‑D float lists.
        batch_size : int, optional
            Number of rows to compute in each batch.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        n_a = len(a)
        n_b = len(b)
        gram = np.empty((n_a, n_b), dtype=np.float64)
        for i in range(0, n_a, batch_size):
            end_i = min(i + batch_size, n_a)
            batch_x = a[i:end_i]
            for j in range(0, n_b, batch_size):
                end_j = min(j + batch_size, n_b)
                batch_y = b[j:end_j]
                sims = np.array(
                    [self([float(x) for x in xx], [float(y) for y in yy]) for xx in batch_x for yy in batch_y]
                ).reshape(end_i - i, end_j - j)
                gram[i:end_i, j:end_j] = sims
        return gram


__all__ = ["QuantumKernelMethod"]
