import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
import numpy as np

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random 2‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumFCL:
    """Parameterized quantum circuit that implements a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        self.theta = theta
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        param_binds = [{self.theta: t} for t in thetas]
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        exp = 0.0
        for state, cnt in counts.items():
            val = 1 if int(state, 2) == 0 else -1
            exp += val * cnt / self.shots
        return np.array([exp])

class QuanvolutionClassifier(nn.Module):
    """Hybrid quantum‑classical classifier that uses a quantum quanvolution filter
    followed by a quantum fully‑connected layer."""
    def __init__(self, n_qubits_fcl: int = 1, shots: int = 1024):
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.fcl = QuantumFCL(n_qubits=n_qubits_fcl, shots=shots)
        self.head = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor, thetas: np.ndarray) -> torch.Tensor:
        features = self.filter(x)  # quantum feature map
        q_out = torch.tensor(self.fcl.run(thetas), device=x.device)
        scale = 1 + q_out
        features = features * scale
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionClassifier"]
