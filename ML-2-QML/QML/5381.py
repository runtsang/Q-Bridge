from __future__ import annotations

import torch
import torchquantum as tq
from typing import Iterable, Tuple, List

# ----------------------------------------------------------------------
# 1. Quantum Quanvolution Filter
# ----------------------------------------------------------------------
class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to patches
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
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
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))

        return torch.cat(patches, dim=1)  # [batch, 196*4]

# ----------------------------------------------------------------------
# 2. Quantum Classifier Ansatz
# ----------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumModule, List[str], List[str], List[str]]:
    """
    Construct a variational classifier ansatz.
    Returns:
        circuit module, list of encoding params, list of weight params, observable names
    """
    encoding = [f"x_{i}" for i in range(num_qubits)]
    weights = [f"theta_{i}" for i in range(num_qubits * depth)]

    # Build the circuit using torchquantum primitives
    circ = tq.QuantumDevice(num_qubits)

    # Encoding
    for q, name in enumerate(encoding):
        circ.rx(name, q)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circ.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circ.cz(q, q + 1)

    # Observables: Z on each qubit
    observables = [f"Z_{q}" for q in range(num_qubits)]
    return circ, encoding, weights, observables

# ----------------------------------------------------------------------
# 3. Quantum Hybrid Model
# ----------------------------------------------------------------------
class QuanvolutionHybridQNN(tq.QuantumModule):
    """
    Quantum counterpart of QuanvolutionHybridNet.
    Performs quantum feature extraction via QuanvolutionFilterQuantum
    followed by a depth‑controlled variational classifier.
    """
    def __init__(self, num_qubits: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum(num_qubits)
        self.classifier_circ, self.enc_params, self.wt_params, self.obs_names = build_classifier_circuit(num_qubits, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 1, 28, 28]
        Returns:
            logits: [batch, num_qubits] from measurement of the classifier circuit
        """
        # Quantum feature extraction
        feats = self.qfilter(x)  # [batch, 196*4]
        # Use first num_qubits features as inputs to the classifier
        batch = feats.shape[0]
        inp = feats[:, :self.enc_params.__len__()]

        # Prepare device for classifier
        qdev = tq.QuantumDevice(self.classifier_circ.n_wires, bsz=batch, device=x.device)

        # Encode inputs
        for name, col in zip(self.enc_params, inp.T):
            qdev.rx(col, int(name.split('_')[-1]))

        # Apply variational layers
        for name in self.wt_params:
            # Random initialization for demonstration; in practice these would be learned
            qdev.ry(torch.randn(batch, 1, device=x.device), int(name.split('_')[-1]))

        # Measurement
        meas = self.measure(qdev)
        return meas  # [batch, num_qubits]

__all__ = [
    "QuanvolutionFilterQuantum",
    "build_classifier_circuit",
    "QuanvolutionHybridQNN",
]
