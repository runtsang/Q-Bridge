import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
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

class QuantumFullyConnectedLayer:
    """Parameterized quantum circuit that maps a vector of rotation angles
    to a 10‑dimensional output by measuring Pauli‑Z on the first ten qubits."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')

    def run(self, thetas: np.ndarray) -> np.ndarray:
        # thetas shape: [B, n_qubits]
        batch_size = thetas.shape[0]
        outputs = []
        for theta in thetas:
            qc = QuantumCircuit(self.n_qubits)
            theta_vec = ParameterVector('theta', self.n_qubits)
            qc.h(range(self.n_qubits))
            qc.ry(theta_vec, range(self.n_qubits))
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            job = execute(qc, self.backend, shots=self.shots,
                          parameter_binds=[{theta_vec[j]: theta[j] for j in range(self.n_qubits)}])
            result = job.result()
            counts = result.get_counts()
            exp_vals = []
            for q in range(self.n_qubits):
                exp = 0.0
                for state, cnt in counts.items():
                    bit = int(state[-(q + 1)])
                    exp += ((-1) ** bit) * cnt
                exp /= self.shots
                exp_vals.append(exp)
            out = np.array(exp_vals[:10])
            if len(out) < 10:
                out = np.pad(out, (0, 10 - len(out)), 'constant')
            outputs.append(out)
        return np.stack(outputs, axis=0)

class QuanvolutionHybrid(nn.Module):
    """Hybrid quantum‑classical model that uses a quantum quanvolution
    filter followed by a parameterised quantum fully‑connected layer.
    The filter encodes image patches into 4‑qubit circuits; the fully
    connected head applies a multi‑qubit rotation circuit and returns
    the expectation values of the first ten qubits as logits."""
    def __init__(self, shots: int = 1024) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = QuantumFullyConnectedLayer(n_qubits=4 * 14 * 14, shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x).detach().cpu().numpy()  # shape [B, 784]
        logits = self.qfc.run(features)                    # shape [B, 10]
        return torch.from_numpy(logits).float()

__all__ = ["QuanvolutionHybrid"]
