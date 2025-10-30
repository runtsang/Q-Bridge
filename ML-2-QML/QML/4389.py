import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Callable, Iterable, List, Sequence

class QuantumNATHybrid(tq.QuantumModule):
    """Quantum‑NAT hybrid model that replaces the classical kernel
    with a variational quantum circuit.  The architecture follows the
    same encoder‑layer‑measure pattern as the original QFCModel while
    adding a downstream classical classifier."""
    class QLayer(tq.QuantumModule):
        """Variational layer that applies a random circuit followed by
        trainable single‑qubit rotations and a controlled‑RX."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            # Optional fixed gates for entanglement
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_classes: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps 16‑dim classical feature vector to qubit states
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Classical classifier head on top of the quantum output
        self.q_to_hidden = nn.Linear(self.n_wires, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical preprocessing identical to the ML version
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Encode classical features
        self.encoder(qdev, pooled)
        # Variational layer
        self.q_layer(qdev)
        # Measure qubits
        qout = self.measure(qdev)
        # Normalise and feed to classical classifier
        qout = self.norm(qout)
        qout = self.q_to_hidden(qout)
        logits = self.classifier(qout)
        return logits

    # ------------------------------------------------------------------
    # Evaluation utilities analogous to FastBaseEstimator
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate observables on batches of inputs using the quantum device."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

__all__ = ["QuantumNATHybrid"]
