"""Enhanced quantum estimator using PennyLane."""

import pennylane as qml
import numpy as np
import torch
from torch.nn.functional import mse_loss
from collections.abc import Iterable, Sequence
from typing import Optional

class FastEstimatorEnhanced:
    """Quantum estimator with a configurable variational ansatz and batched evaluation."""

    def __init__(
        self,
        wires: int | Sequence[int],
        depth: int,
        observables: Iterable[qml.operation.Operation],
        device: Optional[qml.Device] = None,
        shots: Optional[int] = None,
    ) -> None:
        if isinstance(wires, int):
            wires = list(range(wires))
        self.wires = wires
        self.depth = depth
        self.observables = list(observables)
        self.shots = shots
        self.device = device or qml.device("default.qubit", wires=self.wires, shots=self.shots)
        self.qnode = qml.QNode(self._ansatz, self.device, interface="torch")

    def _ansatz(self, params: torch.Tensor):
        """Variational circuit with `depth` layers."""
        n_params = len(self.wires) * 3
        if params.shape[0]!= self.depth * n_params:
            raise ValueError("Parameter vector length mismatch.")
        idx = 0
        for _ in range(self.depth):
            for w in self.wires:
                qml.RX(params[idx], wires=w)
                idx += 1
                qml.RY(params[idx], wires=w)
                idx += 1
                qml.RZ(params[idx], wires=w)
                idx += 1
        return [qml.expval(obs) for obs in self.observables]

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> list[list[float]]:
        """Batch evaluation of expectation values for all parameter sets."""
        param_list = [np.array(p, dtype=np.float32) for p in parameter_sets]
        results = qml.execute(param_list, self.qnode, device=self.device)
        return [list(row) for row in results]

    def train(
        self,
        parameter_sets: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        epochs: int = 10,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Gradient‑based training loop using PyTorch autograd."""
        params = torch.zeros(self.depth * len(self.wires) * 3, requires_grad=True, dtype=torch.float32)
        opt = torch.optim.Adam([params], lr=lr)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        for epoch in range(epochs):
            opt.zero_grad()
            outputs = self.qnode(params)
            loss = mse_loss(torch.tensor(outputs), target_tensor)
            loss.backward()
            opt.step()
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} loss={loss.item():.6f}')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(wires={self.wires}, depth={self.depth})"

__all__ = ["FastEstimatorEnhanced"]
