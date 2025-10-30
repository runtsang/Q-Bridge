"""Quantum fraud‑detection module using PennyLane and optional Qiskit backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import pennylane as qml
import pennylane.numpy as np
import torch

# Optional Qiskit imports
try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.circuit import Parameter
except ImportError:
    QuantumCircuit = None  # type: ignore[assignment]
    Aer = None
    execute = None
    Parameter = None


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic or variational layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    # Additional quantum hyper‑parameters
    entangle: bool = True
    num_layers: int = 1


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def build_fraud_detection_circuit(params: FraudLayerParameters) -> qml.QNode:
    """Return a PennyLane QNode implementing a variational fraud‑detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor) -> torch.Tensor:
        # Feature encoding
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Variational layers
        for _ in range(params.num_layers):
            qml.RZ(params.bs_theta, wires=0)
            qml.RZ(params.bs_phi, wires=1)
            if params.entangle:
                qml.CZ(wires=[0, 1])
            # Optional single‑qubit rotations
            qml.RX(params.squeeze_r[0], wires=0)
            qml.RX(params.squeeze_r[1], wires=1)

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit


def quantum_forward(
    circuit: qml.QNode,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Compute the quantum output for a batch of 2‑dimensional inputs."""
    outputs = []
    for x in inputs:
        outputs.append(circuit(x))
    return torch.stack(outputs)


def hybrid_quantum_loss(
    circuit: qml.QNode,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    alpha: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute a weighted loss between classical labels and quantum predictions."""
    total = 0.0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = quantum_forward(circuit, inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)
        total += loss.item()
    return total / len(data_loader)


def train_quantum(
    circuit: qml.QNode,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    lr: float = 0.01,
    alpha: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Gradient‑based training of the variational circuit."""
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            def loss_fn(weights):
                qml.set_weights(circuit, weights)
                preds = quantum_forward(circuit, inputs)
                return torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)

            opt.step(loss_fn)
            epoch_loss += loss_fn(opt.get_weights(circuit))
        print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(data_loader):.4f}")


def qiskit_fraud_detection_circuit(params: FraudLayerParameters) -> QuantumCircuit:
    """Return a Qiskit circuit equivalent to the PennyLane variational circuit."""
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is not installed")
    qc = QuantumCircuit(2, 2)  # 2 classical bits for measurement
    theta = Parameter("theta")
    phi = Parameter("phi")

    # Feature encoding
    qc.ry(theta, 0)
    qc.ry(phi, 1)

    # Variational layers
    for _ in range(params.num_layers):
        qc.rz(params.bs_theta, 0)
        qc.rz(params.bs_phi, 1)
        if params.entangle:
            qc.cz(0, 1)
        qc.rx(params.squeeze_r[0], 0)
        qc.rx(params.squeeze_r[1], 1)

    # Measurement
    qc.measure([0, 1], [0, 1])
    return qc


def run_qiskit(
    qc: QuantumCircuit,
    inputs: torch.Tensor,
    backend_name: str = "qasm_simulator",
    shots: int = 1024,
) -> torch.Tensor:
    """Execute the Qiskit circuit on a simulator for a batch of inputs."""
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is not installed")
    backend = Aer.get_backend(backend_name)
    results = []
    for x in inputs:
        bound_qc = qc.bind_parameters({"theta": x[0].item(), "phi": x[1].item()})
        job = execute(bound_qc, backend=backend, shots=shots)
        counts = job.result().get_counts()
        # Convert counts to expectation value of Z on qubit 0
        expz = (counts.get("0", 0) - counts.get("1", 0)) / shots
        results.append(expz)
    return torch.tensor(results, dtype=torch.float32)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "quantum_forward",
    "hybrid_quantum_loss",
    "train_quantum",
    "qiskit_fraud_detection_circuit",
    "run_qiskit",
]
