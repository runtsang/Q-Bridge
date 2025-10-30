import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit import ParameterVector

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """2×2 patch quantum kernel applied to image patches."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

def build_qiskit_classifier(num_qubits: int, depth: int):
    """Create a parameterised qiskit circuit for classification."""
    params = ParameterVector('theta', num_qubits * depth)
    qc = qiskit.QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc, params

class UnifiedQuantumNat(tq.QuantumModule):
    """Quantum analogue of the classical UnifiedQuantumNat.
    The model encodes image patches into a small quantum register,
    applies a variational circuit, and classifies via a qiskit
    parameterised classifier.  All trainable parameters are
    exposed as torch Parameters for end‑to‑end optimisation."""
    def __init__(self, num_qubits: int = 4, depth: int = 3, num_classes: int = 10):
        super().__init__()
        self.n_wires = num_qubits
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.quanv = QuantumQuanvolutionFilter(self.n_wires)
        self.vqc = tq.RandomLayer(n_ops=depth, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier_circuit, self.classifier_params = build_qiskit_classifier(self.n_wires, depth)
        self.param_dict = {p: nn.Parameter(torch.rand(1, requires_grad=True)) for p in self.classifier_params}
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        # Dimensionality reduction before encoding
        combined_dim = 16 * 7 * 7 + 4 * 14 * 14
        self.reduce = nn.Linear(combined_dim, self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        cnn_feat = self.cnn(x)
        cnn_flat = cnn_feat.view(bsz, -1)
        quanv_feat = self.quanv(x)
        combined = torch.cat([cnn_flat, quanv_feat], dim=1)
        reduced = self.reduce(combined)
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        for i in range(self.n_wires):
            tq.RX(has_params=True, trainable=False)(qdev, wires=i, params=reduced[:, i])
        self.vqc(qdev)
        measurements = self.measure(qdev)
        logits = []
        shots = 1024
        for sample in measurements.detach().cpu().numpy():
            bound_circ = self.classifier_circuit.bind_parameters(
                {p: self.param_dict[p].item() for p in self.classifier_params}
            )
            job = qiskit.execute(bound_circ, self.backend, shots=shots)
            result = job.result()
            counts = result.get_counts(bound_circ)
            probs = {state: cnt / shots for state, cnt in counts.items()}
            exp = []
            for q in range(self.n_wires):
                exp_q = sum(((-1) ** int(state[-(q + 1)]) * probs[state] for state in probs))
                exp.append(exp_q)
            logits.append(exp)
        return torch.tensor(logits, device=x.device, dtype=torch.float32)

__all__ = ["UnifiedQuantumNat"]
