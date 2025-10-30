"""Hybrid quantum kernel module using TorchQuantum and optional Qiskit EstimatorQNN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# Fraud‑detection layer utilities (quantum analogue)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    q_device: tq.QuantumDevice,
    params: FraudLayerParameters,
    clip: bool = False,
) -> None:
    """Apply a single fraud‑detection layer to the quantum device."""
    # Beam splitter
    tq.ops.BSgate(params.bs_theta, params.bs_phi).on(q_device, (0, 1))
    for i, phase in enumerate(params.phases):
        tq.ops.Rgate(phase).on(q_device, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        tq.ops.Sgate(r if not clip else _clip(r, 5), phi).on(q_device, i)
    # Reverse order for the second half
    tq.ops.BSgate(params.bs_theta, params.bs_phi).on(q_device, (0, 1))
    for i, phase in enumerate(params.phases):
        tq.ops.Rgate(phase).on(q_device, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        tq.ops.Dgate(r if not clip else _clip(r, 5), phi).on(q_device, i)
    for i, k in enumerate(params.kerr):
        tq.ops.Kgate(k if not clip else _clip(k, 1)).on(q_device, i)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> tq.QuantumModule:
    """Create a quantum feature‑mapping module."""
    class FraudCircuit(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.input_params = input_params
            self.layers = list(layers)

        def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
            q_device.reset_states(x.shape[0])
            # Encode input
            for i, wire in enumerate(range(2)):
                tq.ops.RY(x[:, i]).on(q_device, wire)
            # Input layer
            _apply_layer(q_device, self.input_params, clip=False)
            # Hidden layers
            for layer in self.layers:
                _apply_layer(q_device, layer, clip=True)

    return FraudCircuit()


# --------------------------------------------------------------------------- #
# Simple Qiskit EstimatorQNN (for regression)
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> object:
    """Return a Qiskit EstimatorQNN instance."""
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
    from qiskit.primitives import StatevectorEstimator

    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    from qiskit.quantum_info import SparsePauliOp
    observable1 = SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn


# --------------------------------------------------------------------------- #
# Hybrid quantum kernel class
# --------------------------------------------------------------------------- #
class QuantumKernelMethod:
    """
    Quantum kernel engine that supports:
      * Variational quantum feature map (TorchQuantum)
      * Optional fraud‑detection layers
      * Optional Qiskit EstimatorQNN regression
    """

    def __init__(
        self,
        mode: str = "quantum",
        n_wires: int = 4,
        fraud_input: Optional[FraudLayerParameters] = None,
        fraud_layers: Optional[Iterable[FraudLayerParameters]] = None,
        estimator: Optional[object] = None,
    ) -> None:
        if mode!= "quantum":
            raise ValueError("Only 'quantum' mode is supported in this module.")
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.fraud_circuit = build_fraud_detection_program(
            fraud_input, fraud_layers or []
        ) if fraud_input else None
        self.estimator = estimator or EstimatorQNN()

    # ------------------------------------------------------------------ #
    # Feature mapping
    # ------------------------------------------------------------------ #
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x into quantum states and return overlap with reference."""
        if self.fraud_circuit is None:
            # Simple RY encoding
            self.q_device.reset_states(x.shape[0])
            for i in range(x.shape[1]):
                tq.ops.RY(x[:, i]).on(self.q_device, i)
        else:
            self.fraud_circuit(self.q_device, x)
        # Return the first amplitude as a feature (could be generalized)
        return self.q_device.states.view(-1)[0].abs()

    # ------------------------------------------------------------------ #
    # Kernel evaluation
    # ------------------------------------------------------------------ #
    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Overlap of two encoded states."""
        self.transform(x)
        self.transform(y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute Gram matrix for two datasets."""
        return np.array(
            [[self.kernel(x, y).item() for y in b] for x in a]
        )

    # ------------------------------------------------------------------ #
    # Regression interface (Qiskit EstimatorQNN)
    # ------------------------------------------------------------------ #
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 1e-3) -> None:
        """Train the Qiskit EstimatorQNN on the kernel‑mapped features."""
        X_k = torch.stack([self.transform(x) for x in X])
        # Qiskit EstimatorQNN expects numpy arrays
        X_np = X_k.numpy().reshape(-1, 1)
        y_np = y.numpy()
        self.estimator.fit(X_np, y_np, epochs=epochs, lr=lr)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict using the trained estimator on kernel‑mapped features."""
        X_k = torch.stack([self.transform(x) for x in X])
        X_np = X_k.numpy().reshape(-1, 1)
        preds = self.estimator.predict(X_np)
        return torch.tensor(preds, dtype=torch.float32)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "EstimatorQNN",
    "QuantumKernelMethod",
]
