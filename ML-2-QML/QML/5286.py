from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantum as tq
import numpy as np
import qiskit
from qiskit import assemble, transpile

# --------------------------------------------------------------------------- #
# Fraud‑detection style layer – quantum implementation
# --------------------------------------------------------------------------- #
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


class FraudLayerQuantum(tq.QuantumModule):
    """Photonic‑style fraud layer built with TorchQuantum primitives."""
    def __init__(self, params: FraudLayerParameters, clip: bool) -> None:
        super().__init__()
        self.params = params
        self.clip = clip

        # Beam‑splitter
        self.bs1 = tq.BSgate(params.bs_theta, params.bs_phi)
        # Phase rotations
        self.r = [tq.Rgate(phase) for phase in params.phases]
        # Squeezing gates
        self.s = [
            tq.Sgate(r if not clip else _clip(r, 5), phi)
            for r, phi in zip(params.squeeze_r, params.squeeze_phi)
        ]
        # Second beam‑splitter
        self.bs2 = tq.BSgate(params.bs_theta, params.bs_phi)
        # Displacement gates
        self.d = [
            tq.Dgate(r if not clip else _clip(r, 5), phi)
            for r, phi in zip(params.displacement_r, params.displacement_phi)
        ]
        # Kerr gates
        self.k = [
            tq.Kgate(k if not clip else _clip(k, 1)) for k in params.kerr
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.bs1(qdev)
        for gate in self.r: gate(qdev)
        for gate in self.s: gate(qdev)
        self.bs2(qdev)
        for gate in self.r: gate(qdev)
        for gate in self.d: gate(qdev)
        for gate in self.k: gate(qdev)
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# Quantum circuit wrapper (from reference 2)
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


# --------------------------------------------------------------------------- #
# Quantum hybrid head
# --------------------------------------------------------------------------- #
class QuantumHybridHead(tq.QuantumModule):
    """Small quantum circuit that outputs a single expectation value."""
    def __init__(
        self,
        n_qubits: int = 8,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.shift = shift
        self.circuit = QuantumCircuit(n_qubits, self.backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Use the first n_qubits of the input as rotation angles
        angles = x[:, : self.n_qubits].cpu().numpy()
        expectation = self.circuit.run(angles)
        return torch.tensor(expectation, device=x.device).unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Quantum analogue of the classical 2×2 patch extractor."""
    def __init__(self, n_wires: int = 4, patch_size: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.patch_size = patch_size
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz, _, h, w = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, h, w)
        patches = []
        for r in range(0, h, self.patch_size):
            for c in range(0, w, self.patch_size):
                # 4‑pixel patch flattened into 4 inputs
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# Combined quantum network
# --------------------------------------------------------------------------- #
class QuanvolutionClassifier(tq.QuantumModule):
    """Hybrid pipeline: quanvolution filter → fraud‑detection layers → quantum head → linear classifier."""
    def __init__(
        self,
        num_classes: int = 10,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fraud_layers = nn.ModuleList(
            [FraudLayerQuantum(p, clip=True) for p in (fraud_layers or [])]
        )
        self.head = QuantumHybridHead()
        self.classifier = nn.Linear(self.head.n_qubits, num_classes)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        for layer in self.fraud_layers:
            features = layer(features)
        z = self.head(features)
        logits = self.classifier(z)
        return F.log_softmax(logits + self.shift, dim=-1)


__all__ = [
    "FraudLayerParameters",
    "FraudLayerQuantum",
    "QuantumHybridHead",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
]
