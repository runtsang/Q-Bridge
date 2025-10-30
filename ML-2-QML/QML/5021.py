import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List

@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑style fraud detection layer (kept for API consistency)."""
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

class FraudModule(tq.QuantumModule):
    """Encode the photonic fraud‑detection parameters into a quantum device."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        self.params = params
        self.clip = clip
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2x2_ryzxy"])

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.encoder(qdev, torch.tensor([self.params.bs_theta, self.params.bs_phi]))
        # Simplified encoding: use only rotations for demo
        for i, phase in enumerate(self.params.phases):
            tqf.rz(qdev, phase, wires=[i])
        for i, (r, phi) in enumerate(zip(self.params.squeeze_r, self.params.squeeze_phi)):
            tqf.sx(qdev, r if not self.clip else _clip(r, 5), wires=[i])
        for i, (r, phi) in enumerate(zip(self.params.displacement_r, self.params.displacement_phi)):
            tqf.dgate(qdev, r if not self.clip else _clip(r, 5), phi, wires=[i])
        for i, k in enumerate(self.params.kerr):
            tqf.kgate(qdev, k if not self.clip else _clip(k, 1), wires=[i])

class QLayer(tq.QuantumModule):
    """Variational layer that introduces trainable entanglement and rotations."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            getattr(self, f'r{["x","y","z"][wire % 3]}')(qdev, wires=wire)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.cnot(qdev, wires=[3, 0])

class HybridNATNet(tq.QuantumModule):
    """
    Quantum‑NAT hybrid that mirrors the classical architecture while
    replacing the final head with a variational quantum expectation.
    Supports both classification and regression via a simple sigmoid or identity.
    """
    def __init__(
        self,
        in_channels: int = 1,
        task: str = "classification",
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.task = task
        self.n_qubits = n_qubits

        # Classical convolutional backbone (identical to the ML side)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        # Determine flattened size
        dummy = torch.zeros(1, in_channels, 28, 28)
        flat_size = self.features(dummy).view(1, -1).size(1)

        # Optional fraud‑detection quantum module
        if fraud_params:
            # For demo we only encode the first two params into a 2‑wire circuit
            self.fraud = FraudModule(next(iter(fraud_params)), clip=True)
            self.fraud_wires = 2
            self.fraud_out_dim = 1
        else:
            self.fraud = None
            self.fraud_out_dim = 0

        # Quantum variational layer
        self.qlayer = QLayer(self.n_qubits)

        # Measurement (Pauli‑Z on all wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head for classification/regression
        if self.task == "classification":
            self.head = nn.Linear(flat_size + self.fraud_out_dim, 2)
            self.activation = nn.Softmax(dim=-1)
        else:
            self.head = nn.Linear(flat_size + self.fraud_out_dim, 1)
            self.activation = nn.Identity()

        self.bn = nn.BatchNorm1d(self.head.out_features)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical feature extraction
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Encode into quantum device
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device, record_op=False)

        # Simple encoding: use first 4 features as angles
        encoded = x[:, :self.n_qubits]
        self.qlayer.encoder(qdev, encoded)

        # Fraud module if present
        if self.fraud:
            self.fraud(qdev)

        # Variational layer
        self.qlayer(qdev)

        # Measurement
        q_out = self.measure(qdev)  # shape [bsz, n_qubits]
        # Collapse to scalar expectation (mean of Z)
        q_expect = q_out.mean(dim=-1, keepdim=True)

        # Classical head
        out = torch.cat([x, q_expect], dim=-1) if self.fraud_out_dim else x
        out = self.head(out)
        out = self.bn(out)
        return self.activation(out)

    def set_task(self, task: str) -> None:
        """Switch between 'classification' and'regression'."""
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or'regression'")
        self.task = task
        if self.task == "classification":
            self.head = nn.Linear(self.head.in_features, 2)
            self.activation = nn.Softmax(dim=-1)
        else:
            self.head = nn.Linear(self.head.in_features, 1)
            self.activation = nn.Identity()

# Optional helper that returns a Qiskit EstimatorQNN for users who prefer a pure Qiskit backend.
# The function is kept lightweight and only imported if qiskit is available.
try:
    import qiskit
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import StatevectorEstimator
    def get_qiskit_estimator_qnn() -> EstimatorQNN:
        """
        Construct a minimal EstimatorQNN that mimics the variational layer
        used in HybridNATNet.  It can be inserted as a head in place of the
        QuantumModule based hybrid if desired.
        """
        input_param = Parameter("θ")
        weight_param = Parameter("φ")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=estimator,
        )
except Exception:  # pragma: no cover
    get_qiskit_estimator_qnn = None

__all__ = ["HybridNATNet", "FraudLayerParameters", "get_qiskit_estimator_qnn"]
