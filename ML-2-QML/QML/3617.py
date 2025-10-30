import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudTransformerHybrid:
    """
    Quantum‑enhanced hybrid fraud detector.
    Combines photonic fraud‑detection subcircuits with a variational transformer block.
    """
    def __init__(
        self,
        fraud_params: Iterable[FraudLayerParameters],
        seq_len: int,
        n_qubits_per_tx: int = 2,
        device: qml.Device | None = None,
    ) -> None:
        self.fraud_params = list(fraud_params)
        self.seq_len = seq_len
        self.n_qubits = n_qubits_per_tx
        self.total_wires = self.n_qubits * self.seq_len
        self.wires = list(range(self.total_wires))
        self.device = device or qml.device("default.qubit", wires=self.wires)
        self.qnode = qml.QNode(self._circuit, device=self.device, interface="torch")
        # Classical post‑processing head
        self.classifier = nn.Linear(self.total_wires, 1)

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (seq_len, 2) – raw transaction features.
        Returns: Tensor of expectation values of PauliZ on all wires.
        """
        for i in range(self.seq_len):
            wires_i = self.wires[i * self.n_qubits : (i + 1) * self.n_qubits]
            params = self.fraud_params[i]

            # Photonic fraud‑detection subcircuit
            qml.BSgate(params.bs_theta, params.bs_phi, wires=wires_i)
            for j, phase in enumerate(params.phases):
                qml.Rgate(phase, wires=wires_i[j])
            for j, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                qml.Sgate(_clip(r, 5.0), phi, wires=wires_i[j])
            qml.BSgate(params.bs_theta, params.bs_phi, wires=wires_i)
            for j, phase in enumerate(params.phases):
                qml.Rgate(phase, wires=wires_i[j])
            for j, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                qml.Dgate(_clip(r, 5.0), phi, wires=wires_i[j])
            for j, k in enumerate(params.kerr):
                qml.RZ(_clip(k, 1.0), wires=wires_i[j])

        # Variational transformer block – simple entanglement across all wires
        for i in range(self.total_wires):
            qml.RY(np.random.uniform(0, 2 * np.pi), wires=i)
        for i in range(self.total_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        # Circular entanglement
        qml.CNOT(wires=[self.total_wires - 1, 0])

        # Measurement
        return torch.stack([qml.expval(qml.PauliZ(w)) for w in self.wires])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (seq_len, 2) – batch dimension is omitted for simplicity.
        Returns: Tensor of shape (1,) – fraud probability.
        """
        expectations = self.qnode(x)
        return self.classifier(expectations)

__all__ = ["FraudLayerParameters", "FraudTransformerHybrid"]
